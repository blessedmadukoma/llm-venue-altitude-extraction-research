"""
sql_audit.py
============
Diagnostic script for the venue_altitudes pipeline.

Run this at any point to check:
  - Step 0.2  Unmatched venue sample + naming patterns
  - Step 0.2  Fuzzy-match similarity between unmatched and gold venues
  - Step 0.5  Coverage breakdown by source / confidence
  - Step 0.5  Overall join coverage percentage

Usage
-----
    python sql_audit.py

No arguments required.  Reads .env for DB credentials.
"""

import warnings
from db import init_db

warnings.filterwarnings("ignore", category=UserWarning)


# ── helpers ───────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _run(conn, label: str, sql: str, limit: int = 40) -> None:
    """Execute a SQL query and pretty-print the result."""
    _header(label)
    try:
        df = conn.sql(sql).df()
        if df.empty:
            print("  (no rows returned)")
        else:
            print(df.head(limit).to_string(index=False))
            if len(df) > limit:
                print(f"  … {len(df) - limit} more rows not shown")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── Step 0.2 — unmatched venue sample ────────────────────────────────────────

UNMATCHED_SAMPLE_SQL = """
SELECT DISTINCT c.venue, LENGTH(c.venue) AS len
FROM pgdb.crawled_competitions c
LEFT JOIN pgdb.venue_altitudes va
       ON c.venue = va.canonical_venue
       OR c.venue = va.raw_venue_string
WHERE va.canonical_venue IS NULL
  AND c.venue IS NOT NULL
  AND TRIM(c.venue) != ''
ORDER BY c.venue
LIMIT 50
"""

UNMATCHED_COUNT_SQL = """
WITH processed AS (
    SELECT canonical_venue, raw_venue_string FROM pgdb.venue_altitudes
)
SELECT COUNT(DISTINCT c.venue) AS unmatched_venues
FROM pgdb.crawled_competitions c
WHERE c.venue IS NOT NULL
  AND TRIM(c.venue) != ''
  AND c.venue NOT IN (SELECT canonical_venue FROM processed WHERE canonical_venue IS NOT NULL)
  AND c.venue NOT IN (SELECT raw_venue_string FROM processed WHERE raw_venue_string IS NOT NULL)
"""

UNMATCHED_TOP_SQL = """
WITH processed AS (
    SELECT canonical_venue, raw_venue_string FROM pgdb.venue_altitudes
)
SELECT
    c.venue,
    COUNT(DISTINCT c.name) AS n_competitions
FROM pgdb.crawled_competitions c
WHERE c.venue IS NOT NULL
  AND TRIM(c.venue) != ''
  AND c.venue NOT IN (SELECT canonical_venue FROM processed WHERE canonical_venue IS NOT NULL)
  AND c.venue NOT IN (SELECT raw_venue_string FROM processed WHERE raw_venue_string IS NOT NULL)
GROUP BY c.venue
ORDER BY n_competitions DESC
LIMIT 30
"""

# ── Step 0.2 — fuzzy similarity (requires pg_trgm, runs via DuckDB→Postgres) ─

FUZZY_MATCH_SQL = """
SELECT
    c.venue          AS competition_venue,
    va.canonical_venue AS nearest_gold_venue,
    SIMILARITY(c.venue, va.canonical_venue) AS sim_score
FROM (
    SELECT DISTINCT venue FROM pgdb.crawled_competitions
    WHERE venue IS NOT NULL AND TRIM(venue) != ''
) c
CROSS JOIN pgdb.venue_altitudes va
WHERE c.venue NOT IN (
    SELECT canonical_venue FROM pgdb.venue_altitudes WHERE canonical_venue IS NOT NULL
)
  AND c.venue NOT IN (
    SELECT raw_venue_string FROM pgdb.venue_altitudes WHERE raw_venue_string IS NOT NULL
  )
  AND SIMILARITY(c.venue, va.canonical_venue) > 0.4
ORDER BY sim_score DESC
LIMIT 30
"""

# ── Step 0.5 — coverage breakdown ────────────────────────────────────────────

COVERAGE_BREAKDOWN_SQL = """
SELECT
    altitude_source,
    confidence,
    COUNT(*)                                                     AS n_venues,
    ROUND(COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER () * 100, 1)    AS pct
FROM pgdb.venue_altitudes
GROUP BY altitude_source, confidence
ORDER BY altitude_source, confidence
"""

COVERAGE_JOIN_SQL = """
WITH processed AS (
    SELECT canonical_venue, raw_venue_string FROM pgdb.venue_altitudes
)
SELECT
    COUNT(DISTINCT c.venue)  AS total_competition_venues,
    COUNT(DISTINCT CASE
        WHEN c.venue IN (SELECT canonical_venue FROM processed WHERE canonical_venue IS NOT NULL)
          OR c.venue IN (SELECT raw_venue_string FROM processed WHERE raw_venue_string IS NOT NULL)
        THEN c.venue END)    AS matched_venues,
    ROUND(
        100.0 * COUNT(DISTINCT CASE
            WHEN c.venue IN (SELECT canonical_venue FROM processed WHERE canonical_venue IS NOT NULL)
              OR c.venue IN (SELECT raw_venue_string FROM processed WHERE raw_venue_string IS NOT NULL)
            THEN c.venue END)
        / NULLIF(COUNT(DISTINCT c.venue), 0),
    1)                       AS pct_coverage
FROM pgdb.crawled_competitions c
WHERE c.venue IS NOT NULL AND TRIM(c.venue) != ''
"""

ALTITUDE_DISTRIBUTION_SQL = """
SELECT
    CASE
        WHEN altitude_m  < 0    THEN 'below_sea_level'
        WHEN altitude_m  = 0    THEN 'sea_level / default'
        WHEN altitude_m  < 500  THEN '1–499 m'
        WHEN altitude_m  < 1000 THEN '500–999 m'
        WHEN altitude_m  < 2000 THEN '1000–1999 m'
        WHEN altitude_m  < 3000 THEN '2000–2999 m'
        ELSE                         '3000+ m'
    END                        AS altitude_band,
    COUNT(*)                   AS n_venues,
    MIN(altitude_m)            AS min_m,
    MAX(altitude_m)            AS max_m,
    ROUND(AVG(altitude_m), 0)  AS avg_m
FROM pgdb.venue_altitudes
GROUP BY altitude_band
ORDER BY min_m
"""


def main():
    print("\nInitialising DB connection…")
    conn = init_db()

    # ── 0.2 Unmatched stats ───────────────────────────────────────────────────
    _run(conn, "Unmatched venue count", UNMATCHED_COUNT_SQL)
    _run(conn, "Top 30 unmatched venues by competition frequency", UNMATCHED_TOP_SQL)
    _run(conn, "Sample of 50 unmatched venue strings (alphabetical)", UNMATCHED_SAMPLE_SQL)

    # ── 0.2 Fuzzy similarity ─────────────────────────────────────────────────
    print("\n[Fuzzy similarity check — requires pg_trgm extension]")
    print("If this query errors, run:  CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    _run(conn, "Fuzzy matches (sim > 0.4) between unmatched venues and gold", FUZZY_MATCH_SQL)

    # ── 0.5 Coverage breakdown ───────────────────────────────────────────────
    _run(conn, "venue_altitudes rows by source / confidence", COVERAGE_BREAKDOWN_SQL)
    _run(conn, "Overall join coverage vs crawled_competitions", COVERAGE_JOIN_SQL)
    _run(conn, "Altitude distribution across all venue_altitudes rows", ALTITUDE_DISTRIBUTION_SQL)

    try:
        conn.close()
    except Exception:
        pass

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
