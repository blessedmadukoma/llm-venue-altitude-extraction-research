from time import time
import os
import csv
import json
from db import init_db

conn = init_db()

db_table = "crawled_competitions"

# Check how many records are in the competitions table
start = time()
total_records = conn.sql(
    f"SELECT COUNT(*) FROM pgdb.{db_table};").fetchone()[0]
duration = time() - start
print(
    f"Total 'records' in {db_table} table: {total_records}, duration: {duration:.2f}s")
print("================")

# Count total number of unique competitions
start = time()
total_unique_comps = conn.sql(
    f"SELECT COUNT(DISTINCT name) FROM pgdb.{db_table};").fetchone()[0]
duration = time() - start
print(
    f"Total unique 'competitions' in {db_table} table: {total_unique_comps}, duration: {duration:.2f}s")
print("================")

# Count total number of unique venues
start = time()
total_unique_venues = conn.sql(
    f"SELECT COUNT(DISTINCT venue) FROM pgdb.{db_table};").fetchone()[0]
duration = time() - start
print(
    f"Total unique 'venues' in {db_table} table: {total_unique_venues}, duration: {duration:.2f}s")

print("================")
# display first 5 values
start = time()
distinct_venue_comps = conn.sql("""
    WITH cleaned AS (
        SELECT
            venue AS full_venue,
            name AS competition_name,
            date AS competition_date,

            -- Extract country code reliably
            TRIM(regexp_extract(venue, '\\(\\s*([A-Z]{2,3})\\s*\\)', 1)) AS country_code,

            -- Extract the full venue identifier before the country code in parentheses
            -- This is the most consistent key across competitions for the same stadium
            TRIM(regexp_extract(venue, '^\\s*(.+?)\\s*\\([A-Z]{2,3}\\)', 1)) AS venue_key
        FROM pgdb.crawled_competitions
        WHERE regexp_extract(venue, '\\([A-Z]{2,3}\\)', 1) IS NOT NULL
        AND TRIM(regexp_extract(venue, '^\\s*(.+?)\\s*\\([A-Z]{2,3}\\)', 1)) != ''
    ),

    grouped_venues AS (
        SELECT
            -- Use the most frequent full_venue string as the canonical name
            -- (or just ANY_VALUE if you don't care about picking the "best" one)
            mode(full_venue) AS canonical_venue,

            -- All historical mentions for LLM context
            LIST(
                full_venue || ' - ' || competition_name || ' - ' || competition_date::VARCHAR
                ORDER BY competition_date
            ) AS mentions,

            COUNT(*) AS mention_count
        FROM cleaned
        GROUP BY venue_key, country_code
    )

    SELECT
        canonical_venue,
        mention_count,
        mentions
    FROM grouped_venues
    ORDER BY mention_count DESC, canonical_venue;
""").fetchall()

print(f"Length of the dataset: {len(distinct_venue_comps)}")


def pretty_table(headers: list[str], rows: list[list], max_list_items: int = 4):
    """
    Display a clean ASCII table. Handles long lists by showing first few + count.
    """
    if not rows:
        print("No rows to display")
        return

    # Convert values → strings, truncate long lists
    str_rows = []
    for row in rows:
        str_row = []
        for cell in row:
            if cell is None:
                str_row.append("")
            elif isinstance(cell, list):
                if len(cell) > max_list_items:
                    displayed = cell[:max_list_items]
                    summary = f"... (+{len(cell) - max_list_items} more)"
                else:
                    displayed = cell
                    summary = ""
                items = [str(item) for item in displayed]
                if summary:
                    items.append(summary)
                str_row.append("\n".join(items))
            else:
                str_row.append(str(cell))
        str_rows.append(str_row)

    # Calculate column widths (based on longest line in each cell)
    col_widths = []
    for i in range(len(headers)):
        header_width = len(headers[i])
        cell_widths = []
        for row in str_rows:
            lines = row[i].split("\n")
            cell_widths.append(max(len(line) for line in lines))
        col_widths.append(max(header_width, max(cell_widths, default=0)))

    # Build lines
    header_line = " | ".join(headers[i].ljust(
        col_widths[i]) for i in range(len(headers)))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print(header_line)
    print(separator)

    # Print rows with multi-line support
    for str_row in str_rows:
        cell_lines = [str_row[i].split("\n") for i in range(len(headers))]
        max_lines_in_row = max(len(lines)
                               for lines in cell_lines) if cell_lines else 1

        for line_idx in range(max_lines_in_row):
            row_line = []
            for col_idx in range(len(headers)):
                lines = cell_lines[col_idx]
                text = lines[line_idx] if line_idx < len(lines) else ""
                row_line.append(text.ljust(col_widths[col_idx]))
            print(" | ".join(row_line))


# Usage
headers = ["canonical_venue", "mention_count", "mentions (first few + count)"]

# export to csv
out_path = os.path.join(os.path.dirname(__file__),
                        "./data/distinct_venue_comps.csv")
with open(out_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["canonical_venue", "mention_count", "mentions"])
    for canonical, mention_count, mentions in distinct_venue_comps:
        if mentions is None:
            mentions_serialized = ""
        elif isinstance(mentions, (list, tuple)):
            mentions_serialized = json.dumps(mentions, ensure_ascii=False)
        else:
            mentions_serialized = str(mentions)
        writer.writerow(
            [canonical or "", mention_count or 0, mentions_serialized])

print(f"Wrote {len(distinct_venue_comps)} rows to {out_path}")
