"""
rerun_low_confidence.py
=======================
Re-runs LLM inference for venue_altitudes rows that still have poor data:

  - altitude_source = 'default_sea_level'          (LLM previously returned -1)
  - altitude_source = 'satellite' AND confidence IN ('low', 'medium')

For each venue the LLM is called with PROMPT_BULK_TEMPLATE.
The row is updated ONLY if the LLM returns a valid integer altitude (not -1).
Venues where the LLM is still unable to determine altitude are left unchanged.

NOTE: this script intentionally bypasses the satellite-protection guard in
upsert_venue_altitudes() because the user has approved replacing low/medium
confidence satellite rows with LLM data.
"""

import asyncio
import os
import warnings
from typing import Dict

import litellm
import pandas as pd
from dotenv import load_dotenv

from db import _pg_conn, init_db
from llm import BulkGemini
from prompts import PROMPT_BULK_TEMPLATE

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module="litellm.*")
litellm.suppress_debug_info = True

MODEL       = "gemini-3-flash-preview"
CONCURRENCY = 5
LLM_TIMEOUT = 120.0


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_targets(duck_conn) -> pd.DataFrame:
    """Return venue_altitudes rows that need re-inference."""
    return duck_conn.sql("""
        SELECT
            canonical_venue,
            raw_venue_string,
            altitude_m,
            altitude_source,
            confidence
        FROM pgdb.venue_altitudes
        WHERE altitude_source = 'default_sea_level'
           OR (altitude_source = 'satellite' AND confidence IN ('low', 'medium'))
        ORDER BY altitude_source, canonical_venue
    """).df()


def _update_row(canonical_venue: str, altitude_m: int, confidence: str, notes: str) -> bool:
    """
    Directly UPDATE a venue_altitudes row regardless of its current source.
    Returns True if a row was actually changed.
    """
    conn = None
    try:
        conn = _pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE venue_altitudes SET
                    altitude_m      = %s,
                    altitude_source = 'llm_gemini_v4',
                    confidence      = %s,
                    notes           = %s,
                    updated_at      = now()
                WHERE canonical_venue = %s
            """, (altitude_m, confidence, notes, canonical_venue))
            return cur.rowcount > 0
    except Exception as e:
        print(f"  DB error for {canonical_venue[:50]}: {e}")
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _query(llm: BulkGemini, prompt: str) -> Dict:
    try:
        return await asyncio.wait_for(llm.query(prompt), timeout=LLM_TIMEOUT)
    except asyncio.TimeoutError:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}


# ── Processing ────────────────────────────────────────────────────────────────

async def process(venues_df: pd.DataFrame, llm: BulkGemini) -> tuple[int, int]:
    """
    Run inference concurrently and update rows where LLM succeeds.
    Returns (n_updated, n_skipped).
    """
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results   = {"updated": 0, "skipped": 0, "errors": 0}

    async def process_one(row):
        async with semaphore:
            # Prefer raw_venue_string (original scraped name) for the prompt
            raw = (
                row.raw_venue_string
                if pd.notna(row.raw_venue_string) and str(row.raw_venue_string).strip()
                else row.canonical_venue
            )

            result = await _query(llm, PROMPT_BULK_TEMPLATE.replace("{raw_venue}", raw))

            if "error" in result:
                print(f"  ERROR  {row.canonical_venue[:55]}  ({result['error'][:50]})")
                results["errors"] += 1
                return

            altitude = result.get("altitude_meters", -1)
            try:
                altitude = int(altitude)
            except (TypeError, ValueError):
                altitude = -1

            if altitude == -1:
                # LLM couldn't determine — leave the row untouched
                print(f"  SKIP   {row.canonical_venue[:55]}  (undetermined)")
                results["skipped"] += 1
                return

            conf  = str(result.get("confidence", "low")).lower()
            notes = result.get("reasoning", "")
            ok    = _update_row(row.canonical_venue, altitude, conf, notes)

            if ok:
                results["updated"] += 1
                print(f"  OK   {altitude:>6}m  [{conf:<6}]  {row.canonical_venue[:55]}")
            else:
                print(f"  WARN  no row changed for {row.canonical_venue[:50]}")

    tasks = [process_one(row) for row in venues_df.itertuples(index=False)]
    await asyncio.gather(*tasks)

    return results["updated"], results["skipped"]


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("═══════════════════════════════════════════════════════════")
    print(f"  Re-run low-confidence venues  |  Model: {MODEL}")
    print("═══════════════════════════════════════════════════════════\n")

    conn = init_db()

    venues_df = get_targets(conn)
    total = len(venues_df)

    if total == 0:
        print("No low/medium confidence rows found — nothing to do.")
        conn.close()
        return

    # Show breakdown of what we're about to process
    breakdown = (
        venues_df.groupby(["altitude_source", "confidence"])
        .size()
        .reset_index(name="count")
    )
    print(f"Targets: {total} venues\n")
    print(breakdown.to_string(index=False))
    print()

    llm = BulkGemini(model=MODEL)
    n_updated, n_skipped = await process(venues_df, llm)

    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  Updated : {n_updated}")
    print(f"  Skipped : {n_skipped}  (LLM undetermined — left unchanged)")
    print(f"  Remaining unchanged: {total - n_updated - n_skipped}")
    print("─────────────────────────────────────────────────\n")

    try:
        conn.close()
    except Exception:
        pass

    try:
        await litellm.close_litellm_async_clients()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
