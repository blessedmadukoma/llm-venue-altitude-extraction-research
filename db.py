import psycopg2
import psycopg2.extras
import os
import duckdb
from dotenv import load_dotenv
from typing import List, Dict
import pandas as pd

load_dotenv()


def _pg_conn():
    """Return a raw psycopg2 connection using env vars."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def init_postgres_schema():
    """
    DB structure for the results:
    ...
    """
    conn = None
    try:
        conn = _pg_conn()

        with conn, conn.cursor() as cur:
            # Enum
            cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_type WHERE typname = 'confidence_level'
                ) THEN
                    CREATE TYPE confidence_level AS ENUM (
                        'high', 'medium', 'low', 'unknown'
                    );
                END IF;
            END $$;
            """)

            # Gold Standards table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS venues_gold_standard (
                canonical_venue TEXT PRIMARY KEY,

                gold_altitude_m INTEGER NOT NULL,
                gold_confidence confidence_level NOT NULL,
                gold_source TEXT NOT NULL,

                -- country_code VARCHAR(5),
                -- city TEXT NOT NULL,

                mention_count INTEGER NOT NULL DEFAULT 0,
                notes TEXT,

                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                deleted_at TIMESTAMPTZ
            );
            """)

            # Partial index
            cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_venues_gold_standard_active
            ON venues_gold_standard (canonical_venue)
            WHERE deleted_at IS NULL;
            """)

            # Predictions table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS venues_predictions (
                prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                canonical_venue TEXT NOT NULL,

                model_family TEXT NOT NULL,
                model_version TEXT NOT NULL,

                prompt_version TEXT NOT NULL,

                predicted_altitude TEXT NOT NULL,
                confidence_from_model confidence_level NOT NULL,

                reasoning_text TEXT,

                error_absolute INTEGER,
                error_relative_pct NUMERIC(7,3),

                run_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
                run_id TEXT NOT NULL,

                CONSTRAINT fk_venues_predictions_venue
                    FOREIGN KEY (canonical_venue)
                    REFERENCES venues_gold_standard (canonical_venue)
                    ON UPDATE CASCADE
            );
            """)

            # Unified altitude source-of-truth for all venues (gold + LLM)
            cur.execute("""
            CREATE EXTENSION IF NOT EXISTS pg_trgm;
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS venue_altitudes (
                canonical_venue     TEXT        NOT NULL,
                altitude_m          INTEGER     NOT NULL,
                altitude_source     TEXT        NOT NULL,
                confidence          TEXT        NOT NULL,
                error_radius_m      INTEGER,
                raw_venue_string    TEXT,
                notes               TEXT,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

                CONSTRAINT venue_altitudes_pkey PRIMARY KEY (canonical_venue)
            );
            """)

            cur.execute("""
            CREATE INDEX IF NOT EXISTS ix_venue_altitudes_raw
            ON venue_altitudes (raw_venue_string);
            """)

    except Exception as e:
        print("Error initializing Postgres schema:", e)
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def init_db():
    """initialize db connection"""
    print("Initializing database...")

    try:
        init_postgres_schema()
    except Exception:
        print("Aborting DB initialization due to Postgres setup failure.")
        raise

    duck_conn = None
    try:
        duck_conn = duckdb.connect()

        try:
            duck_conn.install_extension("postgres")
            duck_conn.load_extension("postgres")
        except Exception as e:
            # install/load_extension can fail if extension already installed or not available
            print("Warning: problem installing/loading DuckDB postgres extension:", e)

        attach_stmt = f"""
            ATTACH 'dbname={os.getenv("DB_NAME")}
                    user={os.getenv("DB_USER")}
                    password={os.getenv("DB_PASSWORD")}
                    host={os.getenv("DB_HOST")}
                    port={os.getenv("DB_PORT")}'
            AS pgdb (TYPE postgres);
        """

        duck_conn.sql(attach_stmt)
        print()

        return duck_conn

    except Exception as e:
        print("Error initializing DuckDB or attaching Postgres:", e)
        if duck_conn is not None:
            try:
                duck_conn.close()
            except Exception:
                pass
        raise


def store_to_table(table_name: str, columns: List[str], values: pd.DataFrame):
    duck_conn = None
    try:
        duck_conn = init_db()

        table_name = f"pgdb.{table_name}"

        if values is None or values.empty:
            print("No values provided to store; skipping.")
            return

        duck_conn.register("temp", values)

        cols = ", ".join(f'"{c}"' for c in columns)

        duck_conn.sql(f"""
            INSERT INTO {table_name}({cols})
            SELECT {cols} FROM temp;
        """)

    except Exception as e:
        print("Error storing to table:", e)
        raise
    finally:
        if duck_conn is not None:
            try:
                duck_conn.unregister("temp")
            except Exception as e:
                # ignore if 'temp' wasn't registered or unregister fails
                print("Warning: could not unregister 'temp':", e)
            try:
                duck_conn.close()
            except Exception:
                pass


def populate_venue_altitudes_from_gold() -> int:
    """
    Seed venue_altitudes with satellite-verified rows from venues_gold_standard.
    Safe to call multiple times — uses ON CONFLICT DO NOTHING.
    Returns the total number of satellite rows now in the table.
    """
    conn = None
    try:
        conn = _pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO venue_altitudes (
                    canonical_venue, altitude_m, altitude_source,
                    confidence, error_radius_m, raw_venue_string
                )
                SELECT
                    canonical_venue,
                    gold_altitude_m,
                    'satellite',
                    LOWER(gold_confidence::text),
                    16,
                    canonical_venue
                FROM venues_gold_standard
                WHERE deleted_at IS NULL
                  AND gold_altitude_m IS NOT NULL
                ON CONFLICT (canonical_venue) DO NOTHING;
            """)
            cur.execute("""
                SELECT COUNT(*) FROM venue_altitudes
                WHERE altitude_source = 'satellite';
            """)
            count = cur.fetchone()[0]
        print(f"venue_altitudes: {count} satellite rows seeded from gold standard")
        return count
    except Exception as e:
        print("Error seeding venue_altitudes from gold standard:", e)
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def upsert_venue_altitudes(records: List[Dict]) -> int:
    """
    Batch-upsert prediction records into venue_altitudes.
    Never overwrites rows whose altitude_source = 'satellite'.
    Returns the number of rows successfully upserted.
    """
    if not records:
        return 0

    conn = None
    upserted = 0
    try:
        conn = _pg_conn()
        with conn, conn.cursor() as cur:
            for r in records:
                cur.execute("""
                    INSERT INTO venue_altitudes (
                        canonical_venue, altitude_m, altitude_source,
                        confidence, raw_venue_string, notes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (canonical_venue) DO UPDATE SET
                        altitude_m       = EXCLUDED.altitude_m,
                        confidence       = EXCLUDED.confidence,
                        raw_venue_string = EXCLUDED.raw_venue_string,
                        notes            = EXCLUDED.notes,
                        updated_at       = now()
                    WHERE venue_altitudes.altitude_source != 'satellite';
                """, (
                    r["canonical_venue"],
                    r["altitude_m"],
                    r["altitude_source"],
                    r["confidence"],
                    r.get("raw_venue_string", r["canonical_venue"]),
                    r.get("notes", ""),
                ))
                upserted += cur.rowcount
        return upserted
    except Exception as e:
        print("Error upserting venue_altitudes:", e)
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def get_unmatched_venues(duck_conn) -> pd.DataFrame:
    """
    Return venues in crawled_competitions that have no entry in venue_altitudes
    (checked against both canonical_venue and raw_venue_string).
    Ordered by number of competitions descending so high-impact venues are first.
    """
    try:
        df = duck_conn.sql("""
            WITH processed AS (
                SELECT canonical_venue, raw_venue_string
                FROM pgdb.venue_altitudes
            )
            SELECT
                c.venue,
                COUNT(DISTINCT c.name)  AS n_competitions,
                MIN(c.date)             AS first_seen,
                MAX(c.date)             AS last_seen
            FROM pgdb.crawled_competitions c
            WHERE c.venue IS NOT NULL
              AND TRIM(c.venue) != ''
              AND c.venue NOT IN (
                  SELECT canonical_venue FROM processed
                  WHERE canonical_venue IS NOT NULL
              )
              AND c.venue NOT IN (
                  SELECT raw_venue_string FROM processed
                  WHERE raw_venue_string IS NOT NULL
              )
            GROUP BY c.venue
            ORDER BY n_competitions DESC
        """).df()
        return df
    except Exception as e:
        print("Error fetching unmatched venues:", e)
        raise


def venue_altitudes_coverage(duck_conn) -> pd.DataFrame:
    """Return a breakdown of venue_altitudes rows by source and confidence."""
    try:
        df = duck_conn.sql("""
            SELECT
                altitude_source,
                confidence,
                COUNT(*)                                                        AS n_venues,
                ROUND(COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER () * 100, 1)       AS pct
            FROM pgdb.venue_altitudes
            GROUP BY altitude_source, confidence
            ORDER BY altitude_source, confidence
        """).df()
        return df
    except Exception as e:
        print("Error fetching coverage:", e)
        raise


if __name__ == "__main__":
    init_db()
