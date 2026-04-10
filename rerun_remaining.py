"""
rerun_remaining.py
==================
Closes the final join-coverage gap: venues in crawled_competitions that still
have no matching row in venue_altitudes (currently 135).

Root cause of the gap
---------------------
During bulk_inference, when two different raw venue strings (e.g.
"Stadium X (USA)" and "Stadium X, United States") were both normalised by the
LLM to the same canonical name, the second ON CONFLICT DO UPDATE overwrote
raw_venue_string, orphaning the first raw string from the join.

Fix
---
For every still-unmatched raw venue string we INSERT a row whose
canonical_venue = the raw string itself.  This guarantees that the downstream
join  crawled_competitions.venue = venue_altitudes.canonical_venue  always
resolves.  The altitude comes from the LLM when possible; otherwise we store
altitude_m = 0, altitude_source = 'default_sea_level' (safe conservative).

The 6 existing default_sea_level rows (genuinely unresolvable venues) are
intentionally left untouched — this script only targets the 135 that have NO
row in venue_altitudes at all.
"""

import asyncio
import os
import warnings
from typing import Dict

import litellm
import pandas as pd
from dotenv import load_dotenv

from db import _pg_conn, get_unmatched_venues, init_db
from llm import BulkGemini
from prompts import PROMPT_BULK_TEMPLATE

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module="litellm.*")
litellm.suppress_debug_info = True

MODEL       = "gemini-3-flash-preview"
CONCURRENCY = 5
LLM_TIMEOUT = 120.0


# ── DB helper ─────────────────────────────────────────────────────────────────

def insert_with_raw_key(raw_venue: str, altitude_m: int, source: str,
                        confidence: str, notes: str) -> bool:
    """
    INSERT a venue_altitudes row whose PRIMARY KEY is the raw venue string.
    Uses ON CONFLICT DO NOTHING — if for some reason this exact raw string is
    already a canonical_venue, we skip silently.
    """
    conn = None
    try:
        conn = _pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO venue_altitudes (
                    canonical_venue, altitude_m, altitude_source,
                    confidence, raw_venue_string, notes
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (canonical_venue) DO NOTHING;
            """, (raw_venue, altitude_m, source, confidence, raw_venue, notes))
            return cur.rowcount > 0
    except Exception as e:
        print(f"  DB error for {raw_venue[:50]}: {e}")
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
    semaphore = asyncio.Semaphore(CONCURRENCY)
    counts = {"inserted": 0, "sea_level": 0}

    async def process_one(raw_venue: str):
        async with semaphore:
            result = await _query(llm, PROMPT_BULK_TEMPLATE.replace("{raw_venue}", raw_venue))

            if "error" in result:
                print(f"  ERROR  {raw_venue[:60]}  ({result['error'][:40]})")
                # Store as sea level so the join works even on failure
                ok = insert_with_raw_key(raw_venue, 0, "default_sea_level",
                                         "low", f"LLM error: {result['error']}")
                if ok:
                    counts["sea_level"] += 1
                return

            altitude = result.get("altitude_meters", -1)
            try:
                altitude = int(altitude)
            except (TypeError, ValueError):
                altitude = -1

            if altitude == -1:
                print(f"  ??     {raw_venue[:60]}  (undetermined → sea level)")
                ok = insert_with_raw_key(raw_venue, 0, "default_sea_level",
                                         "low", result.get("reasoning", ""))
                if ok:
                    counts["sea_level"] += 1
                return

            conf  = str(result.get("confidence", "low")).lower()
            notes = result.get("reasoning", "")
            ok    = insert_with_raw_key(raw_venue, altitude, "llm_gemini_v4", conf, notes)
            if ok:
                counts["inserted"] += 1
            print(f"  OK   {altitude:>6}m  [{conf:<6}]  {raw_venue[:60]}")

    tasks = [process_one(row.venue) for row in venues_df.itertuples(index=False)]
    await asyncio.gather(*tasks)
    return counts["inserted"], counts["sea_level"]


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("═══════════════════════════════════════════════════════════")
    print(f"  Fill remaining join-coverage gap  |  Model: {MODEL}")
    print("═══════════════════════════════════════════════════════════\n")

    conn = init_db()

    venues_df = get_unmatched_venues(conn)
    total = len(venues_df)

    if total == 0:
        print("No unmatched venues found — join coverage is already 100%.")
        conn.close()
        return

    print(f"Unmatched venues: {total}\n")

    llm = BulkGemini(model=MODEL)
    n_inserted, n_sea_level = await process(venues_df, llm)

    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  Inserted (llm_gemini_v4)  : {n_inserted}")
    print(f"  Inserted (default/unknown): {n_sea_level}")
    print(f"  Total inserted            : {n_inserted + n_sea_level} / {total}")
    print("─────────────────────────────────────────────────")
    print("\nRe-run sql_audit.py to verify final coverage.")

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
