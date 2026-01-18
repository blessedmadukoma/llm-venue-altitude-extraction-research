"""Analyze the results from the db"""
import pandas as pd
import warnings
from db import init_db
from datetime import datetime
import os

warnings.filterwarnings("ignore", category=UserWarning)

# OpenRouter CSV costs file
DETAILED_CSV_PATH = "./data/openrouter_activity_2026-01-15.csv"

MODEL_MAPPING = {
    'gpt-5.2':        'openai/gpt-5.2-20251211',
    'gemini-2.5-pro': 'google/gemini-2.5-pro',
    'claude-4.5-sonnet': 'anthropic/claude-4.5-sonnet-20250929',
    'qwen3-8b':       'qwen/qwen3-8b-04-28'
}


def load_predictions_from_db():
    """Load recent predictions from your venues_predictions table"""
    try:
        conn = init_db()
        query = """
        SELECT 
            canonical_venue,
            model_family,
            model_version,
            prompt_version,
            predicted_altitude,
            confidence_from_model,
            reasoning_text,
            run_timestamp,
            run_id
        FROM pgdb.venues_predictions
        WHERE run_timestamp >= '2026-01-01'
        ORDER BY run_timestamp
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("No predictions found in the database for the selected period.")
            return None

        # Clean predicted_altitude (it's TEXT)
        df['predicted_altitude_num'] = pd.to_numeric(
            df['predicted_altitude'], errors='coerce')
        df['is_unknown'] = df['predicted_altitude'].str.lower().str.contains(
            'unknown|error|n/a|null|parse failed', na=True, regex=True
        )

        print(f"Loaded {len(df)} prediction rows from database.")
        return df

    except Exception as e:
        print(f"Error loading predictions from DB: {e}")
        return None


def load_openrouter_costs(detailed_csv_path):
    """Load and prepare the detailed OpenRouter CSV"""
    try:
        df = pd.read_csv(detailed_csv_path)
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)

        if 'api_key_name' in df.columns:
            df = df[df['api_key_name'].str.contains(
                'cmu-msc-sport-research', na=False)]

        # Filter to the main run time window (05:00–07:00 UTC on 2026-01-15)
        df = df[
            (df['created_at'].dt.date == pd.Timestamp('2026-01-15').date()) &
            (df['created_at'].dt.hour >= 5) &
            (df['created_at'].dt.hour < 7)
        ]

        df = df.sort_values(
            by="created_at", ascending=True).reset_index(drop=True)

        print(
            f"Loaded and filtered {len(df)} cost records from CSV (05:00–07:00 UTC).")
        return df

    except Exception as e:
        print(f"Error loading OpenRouter CSV: {e}")
        return None


def match_costs_to_predictions(pred_df: pd.DataFrame, costs_df: pd.DataFrame):
    """
    Match predictions to costs using:
    1. Group by model_key
    2. Sort each group by timestamp
    3. Row-by-row alignment within each group
    """
    if pred_df is None or costs_df is None or pred_df.empty or costs_df.empty:
        print("Cannot match: empty input")
        return None

    pred = pred_df.copy()
    costs = costs_df.copy()

    # Ensure UTC datetime
    pred['run_timestamp'] = pd.to_datetime(pred['run_timestamp'], utc=True)
    costs['created_at'] = pd.to_datetime(costs['created_at'], utc=True)

    # Normalize model_key
    def normalize_model(name):
        name = str(name).lower().strip()
        if 'gpt-5.2' in name:
            return 'gpt-5.2'
        if 'gemini-2.5-pro' in name:
            return 'gemini-2.5-pro'
        if 'claude-4.5-sonnet' in name:
            return 'claude-4.5-sonnet'
        if 'qwen3-8b' in name:
            return 'qwen3-8b'
        return 'unknown'

    pred['model_key'] = pred['model_family'].apply(normalize_model)
    costs['model_key'] = costs['model_permaslug'].apply(normalize_model)

    # Drop unknowns
    pred = pred[pred['model_key'] != 'unknown']
    costs = costs[costs['model_key'] != 'unknown']

    print(f"Predictions after cleaning: {len(pred)} rows")
    print(f"Costs after cleaning:       {len(costs)} rows")

    # Get unique model_keys
    model_keys = pred['model_key'].unique()

    merged_groups = []

    for key in model_keys:
        print(f"\nProcessing group: {key}")

        pred_group = pred[pred['model_key'] == key].sort_values(
            'run_timestamp').reset_index(drop=True)
        costs_group = costs[costs['model_key'] == key].sort_values(
            'created_at').reset_index(drop=True)

        print(f"  Pred rows for {key}:   {len(pred_group)}")
        print(f"  Costs rows for {key}:  {len(costs_group)}")

        if len(pred_group) != len(costs_group):
            print(
                f"  Warning: Row count mismatch for {key} — using left join (unmatched will have NaN costs)")

        # Row-by-row concat (aligned by sorted order)
        group_merged = pd.concat([pred_group, costs_group], axis=1)

        # Diagnostic: time diff
        group_merged['time_diff_sec'] = (
            group_merged['created_at'] - group_merged['run_timestamp']).dt.total_seconds().abs()

        # Keep only relevant cost columns (rename for clarity)
        group_merged = group_merged.rename(columns={
            'cost_total': 'cost_usd',
            'tokens_prompt': 'prompt_tokens',
            'tokens_completion': 'completion_tokens',
            'tokens_reasoning': 'reasoning_tokens'
        })

        # Summary per group
        matched = group_merged['cost_usd'].notna().sum()
        print(
            f"  Matched: {matched} / {len(group_merged)} ({matched/len(group_merged):.1%})")
        print(f"  Avg time diff: {group_merged['time_diff_sec'].mean():.1f} s")
        print(f"  Max time diff: {group_merged['time_diff_sec'].max():.1f} s")

        merged_groups.append(group_merged)

    # Combine all groups
    final_merged = pd.concat(merged_groups, ignore_index=True)

    # Global summary
    global_matched = final_merged['cost_usd'].notna().sum()
    print(
        f"\nGlobal match rate: {global_matched} / {len(final_merged)} ({global_matched/len(final_merged):.1%})")

    # Show sample mismatches if any
    mismatches = final_merged[final_merged['cost_usd'].isna()]
    if not mismatches.empty:
        print(f"\n{len(mismatches)} unmatched rows (first 5):")
        print(mismatches[['canonical_venue', 'model_key',
              'prompt_version', 'run_timestamp']].head())

    return final_merged


def compute_metrics(merged_df, gold_df):
    """Compute accuracy metrics using gold standard elevations"""
    if merged_df is None or gold_df is None:
        return None

    # Join with gold elevations
    gold_subset = gold_df[['canonical_venue', 'elevation']].drop_duplicates()
    merged = merged_df.merge(gold_subset, on='canonical_venue', how='left')

    # Valid predictions only
    valid = merged[
        (~merged['is_unknown']) &
        merged['predicted_altitude_num'].notnull() &
        merged['elevation'].notnull()
    ].copy()

    if valid.empty:
        print("No valid numeric predictions to evaluate.")
        return None

    valid['error_absolute'] = abs(
        valid['predicted_altitude_num'] - valid['elevation'])
    valid['is_large_error'] = valid['error_absolute'] > 100

    # Group metrics
    metrics = valid.groupby(['model_family', 'prompt_version']).agg(
        count_valid=('canonical_venue', 'count'),
        mae=('error_absolute', 'mean'),
        median_ae=('error_absolute', 'median'),
        pct_large_errors=('is_large_error', 'mean')
    ).reset_index()

    # Unknowns percentage
    unknowns = merged.groupby(['model_family', 'prompt_version']).agg(
        total_queries=('canonical_venue', 'count'),
        pct_unknown=('is_unknown', 'mean')
    ).reset_index()

    costs = merged.groupby(['model_family', 'prompt_version']).agg(
        avg_cost_usd=('cost_usd', 'mean'),
        total_cost_usd=('cost_usd', 'sum'),
        avg_prompt_tokens=('prompt_tokens', 'mean'),
        avg_completion_tokens=('completion_tokens', 'mean'),
        avg_reasoning_tokens=('reasoning_tokens', 'mean')
    ).reset_index()

    # Combine all
    results = metrics.merge(
        unknowns, on=['model_family', 'prompt_version'], how='left')
    results = results.merge(
        costs, on=['model_family', 'prompt_version'], how='left')

    # Fill missing costs with 0
    results['avg_cost_usd'] = results['avg_cost_usd'].fillna(0)
    results['total_cost_usd'] = results['total_cost_usd'].fillna(0)

    return results


def rank_methods(results_df):
    """Create a ranking score and sort"""
    if results_df is None or results_df.empty:
        return None

    # Normalize and combine (lower is better)
    results_df = results_df.copy()

    # Scale components roughly to similar magnitude
    results_df['score_mae'] = results_df['mae'].fillna(1000)
    results_df['score_large'] = results_df['pct_large_errors'].fillna(1) * 100
    results_df['score_unknown'] = results_df['pct_unknown'].fillna(1) * 100
    results_df['score_cost'] = results_df['avg_cost_usd'].fillna(
        0) * 5000  # $0.001 → 5 points

    results_df['combined_score'] = (
        results_df['score_mae'] +
        results_df['score_large'] +
        results_df['score_unknown'] +
        results_df['score_cost']
    )

    ranked = results_df.sort_values('combined_score').reset_index(drop=True)
    ranked['rank'] = ranked.index + 1

    return ranked


if __name__ == "__main__":
    print("=== Altitude Prediction Analysis with Real OpenRouter Costs ===\n")

    # load gold standard (adjust path & column name if needed)
    try:
        gold_venues = pd.read_csv("./data/gold_venues_with_elevation.csv")
        print(f"Gold standard venues loaded: {len(gold_venues)} rows")
    except Exception as e:
        print(f"Error loading gold CSV: {e}")
        gold_venues = None

    # load predictions & costs
    predictions = load_predictions_from_db()
    costs_df = load_openrouter_costs(DETAILED_CSV_PATH)

    # match costs to predictions
    matched_df = match_costs_to_predictions(predictions, costs_df)

    # Compute metrics
    results = compute_metrics(matched_df, gold_venues)

    if results is not None:
        ranked = rank_methods(results)

        if ranked is not None:
            print("\nModel / Prompt Performance Ranking (lower score = better)\n")
            display_cols = [
                'rank', 'model_family', 'prompt_version',
                'mae', 'pct_large_errors', 'pct_unknown',
                'avg_cost_usd', 'total_cost_usd',
                'count_valid', 'combined_score'
            ]
            print(ranked[display_cols].round(4).to_string(index=False))

            print("\nRecommended winner:")
            winner = ranked.iloc[0]
            print(f"→ {winner['model_family']} - {winner['prompt_version']}")
            print(f"   MAE: {winner['mae']:.2f}m")
            print(f"   Large errors (>100m): {winner['pct_large_errors']:.1%}")
            print(f"   Unknown rate: {winner['pct_unknown']:.1%}")
            print(f"   Avg cost per query: ${winner['avg_cost_usd']:.6f}")
            print(f"   Combined score: {winner['combined_score']:.2f}")

            # Export textual results to a .txt file
            out_path = os.path.join("output", "analysis_results.txt")
            # ensure output directory exists
            out_dir = os.path.dirname(out_path)
            if out_dir:
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception as e:
                    print(f"Could not create output directory {out_dir}: {e}")

            lines = []
            lines.append("Altitude Prediction Analysis Results")
            lines.append(f"Generated (UTC): {datetime.utcnow().isoformat()}")
            lines.append("")

            if 'ranked' in globals() and ranked is not None and not ranked.empty:
                lines.append(
                    "Model / Prompt Performance Ranking (lower score = better)")
                display_cols = [
                    'rank', 'model_family', 'prompt_version',
                    'mae', 'pct_large_errors', 'pct_unknown',
                    'avg_cost_usd', 'total_cost_usd',
                    'count_valid', 'combined_score'
                ]
                try:
                    ranking_text = ranked[display_cols].round(
                        6).to_string(index=False)
                except Exception:
                    ranking_text = ranked.to_string(index=False)
                lines.append(ranking_text)
                lines.append("")

                winner = ranked.iloc[0]
                mae = float(winner.get('mae', float('nan')))
                pct_large = float(winner.get('pct_large_errors', 0))
                pct_unknown = float(winner.get('pct_unknown', 0))
                avg_cost = float(winner.get('avg_cost_usd', 0))
                combined = float(winner.get('combined_score', 0))

                lines.append("Recommended winner:")
                lines.append(
                    f"→ {winner['model_family']} - {winner['prompt_version']}")
                lines.append(f"   MAE: {mae:.2f}m")
                lines.append(f"   Large errors (>100m): {pct_large:.1%}")
                lines.append(f"   Unknown rate: {pct_unknown:.1%}")
                lines.append(f"   Avg cost per query: ${avg_cost:.6f}")
                lines.append(f"   Combined score: {combined:.2f}")

            else:
                if 'results' in globals() and results is not None and not results.empty:
                    lines.append(
                        "No ranking available; writing raw results table instead.")
                    lines.append(results.round(6).to_string(index=False))
                else:
                    lines.append("No results to export.")

            # Write the results file
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print(f"\nExported textual results to: {out_path}")
            except Exception as e:
                print(f"Failed to write results to {out_path}: {e}")
        else:
            print("Could not create ranking.")
    else:
        print("Analysis could not be completed — check data loading.")
