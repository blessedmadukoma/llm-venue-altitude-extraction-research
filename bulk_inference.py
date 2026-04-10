"""
bulk_inference.py
=================
Populates venue_altitudes for all ~4k competition venues that have no altitude
data yet.  Uses gemini-2.5-pro + PROMPT_BULK_TEMPLATE (best-performing combo).

Flow
----
1.  Init DB schema (creates venue_altitudes if missing).
2.  Seed venue_altitudes from gold standard (satellite data, ~145 venues).
3.  Query crawled_competitions for venues not yet covered → venues_to_predict.csv
4.  Run LLM inference in async batches (CONCURRENCY parallel requests).
5.  After every CHUNK_SIZE venues, upsert results to DB + append to CSV.
6.  Print coverage report on completion.

Resume safety
-------------
Re-running the script is safe: get_unmatched_venues() skips venues already in
venue_altitudes, and upsert_venue_altitudes() uses ON CONFLICT DO UPDATE but
never overwrites satellite data.

Sentinel value
--------------
LLM returns altitude_meters = -1 when it cannot determine the altitude.
These are stored as altitude_m = 0, altitude_source = 'default_sea_level',
confidence = 'low'.  They receive no altitude correction downstream, which
is the conservative/safe behaviour.
"""

import asyncio
import csv
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import litellm
import pandas as pd
from dotenv import load_dotenv

from db import (
    get_unmatched_venues,
    init_db,
    populate_venue_altitudes_from_gold,
    upsert_venue_altitudes,
    venue_altitudes_coverage,
)
from llm import BulkGemini
from prompts import PROMPT_BULK_TEMPLATE

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module="litellm.*")

# Suppress litellm's "Provider List:" banner that prints on every request
litellm.suppress_debug_info = True

# ── tuneable constants ────────────────────────────────────────────────────────
CONCURRENCY = 10      # parallel LLM requests
CHUNK_SIZE = 50      # upsert + CSV flush every N venues
LLM_TIMEOUT = 120.0   # seconds before a single call is abandoned
RESULTS_CSV = "./data/bulk_inference_results.csv"
VENUES_CSV = "./data/venues_to_predict.csv"
# ─────────────────────────────────────────────────────────────────────────────


async def _query_with_timeout(llm: BulkGemini, prompt: str) -> Dict:
    """Single LLM call wrapped with a timeout."""
    try:
        return await asyncio.wait_for(llm.query(prompt), timeout=LLM_TIMEOUT)
    except asyncio.TimeoutError:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}


def _build_record(raw_venue: str, result: Dict) -> Dict:
    """
    Convert a raw LLM result dict into a venue_altitudes-compatible record.
    Handles errors, -1 sentinel, and type coercion.
    """
    if "error" in result:
        return {
            "raw_venue_string": raw_venue,
            "canonical_venue":  raw_venue,
            "altitude_m":       0,
            "altitude_source":  "default_sea_level",
            "confidence":       "low",
            "notes":            f"LLM error: {result['error']}",
        }

    altitude = result.get("altitude_meters", -1)
    try:
        altitude = int(altitude)
    except (TypeError, ValueError):
        altitude = -1

    if altitude == -1:
        return {
            "raw_venue_string": raw_venue,
            "canonical_venue":  result.get("canonical_venue", raw_venue),
            "altitude_m":       0,
            "altitude_source":  "default_sea_level",
            "confidence":       "low",
            "notes":            f"LLM undetermined. {result.get('reasoning', '')}",
        }

    return {
        "raw_venue_string": raw_venue,
        "canonical_venue":  result.get("canonical_venue", raw_venue),
        "altitude_m":       altitude,
        "altitude_source":  "llm_gemini_v4",
        "confidence":       str(result.get("confidence", "low")).lower(),
        "notes":            result.get("reasoning", ""),
    }


def _append_to_csv(records: List[Dict], path: str, write_header: bool) -> None:
    """Append a list of record dicts to a CSV file."""
    if not records:
        return
    fieldnames = ["raw_venue_string", "canonical_venue", "altitude_m",
                  "altitude_source", "confidence", "notes"]
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


async def process_venues(venues_df: pd.DataFrame, llm: BulkGemini) -> List[Dict]:
    """
    Process all venues asynchronously.
    Upserts to DB and flushes to CSV after every CHUNK_SIZE completions.
    Returns the full list of result records.
    """
    total = len(venues_df)
    all_records: List[Dict] = []
    semaphore = asyncio.Semaphore(CONCURRENCY)
    write_header = not os.path.exists(RESULTS_CSV)

    async def process_one(raw_venue: str) -> Optional[Dict]:
        async with semaphore:
            # Build prompt — use str.replace to avoid issues with curly braces
            # in venue names and the double-brace escaping in the template.
            prompt = PROMPT_BULK_TEMPLATE.replace("{raw_venue}", raw_venue)
            result = await _query_with_timeout(llm, prompt)
            record = _build_record(raw_venue, result)
            src = record["altitude_source"]
            alt = record["altitude_m"]
            conf = record["confidence"]
            print(f"  {'OK' if src != 'default_sea_level' else '??'}  "
                  f"{alt:>5}m  [{conf:<6}]  {raw_venue[:60]}")
            return record

    # Build all coroutine tasks upfront
    tasks = [process_one(row.venue)
             for row in venues_df.itertuples(index=False)]

    # Process in chunks so we save progress frequently
    processed = 0
    for chunk_start in range(0, len(tasks), CHUNK_SIZE):
        chunk_tasks = tasks[chunk_start: chunk_start + CHUNK_SIZE]
        chunk_venues = list(
            venues_df["venue"].iloc[chunk_start: chunk_start + CHUNK_SIZE])
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        chunk_records: List[Dict] = []
        for venue, res in zip(chunk_venues, chunk_results):
            if isinstance(res, Exception):
                print(f"  FAIL  {venue[:60]}  ({res})")
                chunk_records.append(_build_record(venue, {"error": str(res)}))
            elif res is not None:
                chunk_records.append(res)

        if chunk_records:
            upserted = upsert_venue_altitudes(chunk_records)
            _append_to_csv(chunk_records, RESULTS_CSV,
                           write_header=write_header)
            write_header = False
            all_records.extend(chunk_records)

        processed += len(chunk_tasks)
        print(f"\n── Chunk done: {processed}/{total} venues processed "
              f"({processed/total*100:.1f}%)  "
              f"[{datetime.utcnow().strftime('%H:%M:%S')} UTC] ──\n")

    return all_records


def print_coverage_report(duck_conn) -> None:
    """Print source/confidence breakdown and overall join coverage."""
    print("\n═══════════════════════════════════════════════════════════")
    print("  venue_altitudes  coverage report")
    print("═══════════════════════════════════════════════════════════")

    try:
        cov_df = venue_altitudes_coverage(duck_conn)
        if not cov_df.empty:
            print(cov_df.to_string(index=False))
    except Exception as e:
        print(f"Coverage breakdown error: {e}")

    try:
        join_df = duck_conn.sql("""
            WITH processed AS (
                SELECT canonical_venue, raw_venue_string FROM pgdb.venue_altitudes
            )
            SELECT
                COUNT(DISTINCT c.venue)  AS total_competition_venues,
                COUNT(DISTINCT CASE
                    WHEN c.venue IN (SELECT canonical_venue FROM processed
                                     WHERE canonical_venue IS NOT NULL)
                      OR c.venue IN (SELECT raw_venue_string FROM processed
                                     WHERE raw_venue_string IS NOT NULL)
                    THEN c.venue END)    AS matched_venues
            FROM pgdb.crawled_competitions c
            WHERE c.venue IS NOT NULL AND TRIM(c.venue) != ''
        """).df()

        total = int(join_df["total_competition_venues"].iloc[0])
        matched = int(join_df["matched_venues"].iloc[0])
        pct = matched / total * 100 if total else 0
        print(f"\nJoin coverage: {matched}/{total} venues  ({pct:.1f}%)")
    except Exception as e:
        print(f"Join coverage query error: {e}")

    print("═══════════════════════════════════════════════════════════\n")


async def main():
    model = "gemini-3-flash-preview"
    print("═══════════════════════════════════════════════════════════")
    print("  Altitude Bulk Inference Pipeline")
    print(f"  Model: {model}  |  Prompt: BULK_TEMPLATE")
    print("═══════════════════════════════════════════════════════════\n")

    conn = None
    try:
        # Step 1: Init DB schema
        print("Step 1 — Initialising DB schema...")
        conn = init_db()
        print()

        # Step 2: Seed venue_altitudes from gold standard
        print("Step 2 — Seeding venue_altitudes from gold standard...")
        populate_venue_altitudes_from_gold()
        print()

        # Step 3: Find venues with no altitude data
        print("Step 3 — Querying unmatched venues...")
        venues_df = get_unmatched_venues(conn)
        total = len(venues_df)
        print(f"         {total} venues need altitude inference")

        if total == 0:
            print("\nAll venues already have altitude data. Nothing to do.")
            print_coverage_report(conn)
            return

        # Save venue list for audit trail
        venues_df.to_csv(VENUES_CSV, index=False)
        print(f"         Venue list saved to {VENUES_CSV}\n")

        # Step 4: LLM inference
        print(f"Step 4 — Running LLM inference (concurrency={CONCURRENCY}, "
              f"chunk_size={CHUNK_SIZE})...")
        print(f"         Results streaming to {RESULTS_CSV}\n")

        llm = BulkGemini(model=model)

        wall_start = time.perf_counter()
        all_records = await process_venues(venues_df, llm)
        wall_elapsed = time.perf_counter() - wall_start

        print(f"Step 4 done — {len(all_records)} records processed "
              f"in {wall_elapsed/60:.1f} min")

        # Step 5: Coverage report
        print_coverage_report(conn)

    except Exception as e:
        print(f"\nFatal error: {e}")
        raise

    finally:
        try:
            await litellm.close_litellm_async_clients()
        except Exception:
            pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        print("DB connection closed.")


if __name__ == "__main__":
    _t0 = time.perf_counter()
    asyncio.run(main())
    _elapsed = time.perf_counter() - _t0
    _h, _r = divmod(_elapsed, 3600)
    _m, _s = divmod(_r, 60)
    print(f"\nTotal runtime: {int(_h)}h {int(_m)}m {_s:.2f}s")
