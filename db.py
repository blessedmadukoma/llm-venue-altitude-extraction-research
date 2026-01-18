import psycopg2
import os
import duckdb
from dotenv import load_dotenv
from typing import List
import pandas as pd

load_dotenv()


def init_postgres_schema():
    """
    DB structure for the results:
    ...
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )

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


if __name__ == "__main__":
    init_db()
