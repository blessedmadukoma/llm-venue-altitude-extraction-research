# Altitude Integration Plan — Adjusted Status

## Context

This plan integrates venue altitude data into the Athletics Anomaly Detection Pipeline.
**Phase 0 (data foundation) is fully complete.** The remaining work (Phases 1–6) is
app-side integration: wiring the `venue_altitudes` PostgreSQL table into the detection
pipeline so that altitude corrections are applied before statistical/ML detectors run.

---

## ✅ Phase 0 — Data Foundation (COMPLETE)

**What was built (in `altitude-research/`):**

| Item | Status | Detail |
|------|--------|--------|
| `venue_altitudes` PostgreSQL table | ✅ Done | Single source of truth; PK = `canonical_venue` |
| Schema guard | ✅ Done | `ON CONFLICT DO UPDATE WHERE altitude_source != 'satellite'` — never overwrites satellite rows |
| Gold-standard seed | ✅ Done | 145 satellite-verified venues loaded via `populate_venue_altitudes_from_gold()` |
| Bulk LLM inference | ✅ Done | `bulk_inference.py` — 4,135 venues processed with `BulkGemini` (gemini-3-flash-preview) |
| Low-confidence re-run | ✅ Done | `rerun_low_confidence.py` — 58 default_sea_level rows re-inferred; 6 genuinely unresolvable remain |
| Join-gap closure | ✅ Done | `rerun_remaining.py` — inserts orphaned raw strings as canonical_venue PK |
| Diagnostic tooling | ✅ Done | `sql_audit.py` for coverage/confidence breakdown |

**Final `venue_altitudes` row breakdown:**

| altitude_source | confidence | n_venues |
|-----------------|------------|----------|
| satellite | high | 115 |
| llm_gemini_v4 | high | ~3,992 |
| llm_gemini_v4 | medium | ~32 |
| default_sea_level | low | 6 |

**Join coverage: 4,145 / 4,280 competition venues (96.8%+)**

The 6 default_sea_level rows are genuinely unresolvable (e.g., "TBC", blank-ish venue
strings). They receive no altitude correction downstream — conservative/safe behaviour.

---

## Phase 1 — DuckDB View + Data Loader

**Files to modify (in the app project):**

### 1.1 `app/db/duckdb_client.py`

Add a DuckDB view that exposes `venue_altitudes` via the `pgdb` attachment:

```python
conn.execute("""
    CREATE OR REPLACE VIEW venue_altitudes AS
    SELECT * FROM pgdb.venue_altitudes
""")
```

Update existing `wind_legal_results` and `all_results` views to LEFT JOIN altitude:

```sql
LEFT JOIN pgdb.venue_altitudes va
       ON c.venue = va.canonical_venue
       OR c.venue = va.raw_venue_string
```

> ⚠️ The `OR c.venue = va.raw_venue_string` clause is essential — it covers the orphaned
> raw strings inserted by `rerun_remaining.py`.

### 1.2 `app/core/data_loader.py`

Add altitude columns to the SELECT that feeds the DataFrame:

```python
"va.altitude_m",
"va.altitude_source",
"va.confidence AS altitude_confidence",
"va.canonical_venue AS venue_name",
```

Fill NaN with 0 and `'unknown'` for the ~3.2% unmatched venues.

---

## Phase 2 — Altitude Correction Module

**New file: `app/core/altitude.py`**

Event-specific corrections based on World Athletics altitude adjustment tables:

```python
# Correction coefficients (seconds per 1000 m altitude), by event
ALTITUDE_CORRECTIONS = {
    "100m":    0.03,
    "200m":    0.07,
    "400m":    0.17,
    "800m":   -0.14,   # aerobic demand dominates — middle-distance penalised
    "1500m":  -0.50,
    "5000m":  -2.00,
    "10000m": -4.00,
    "marathon": -45,
}

def sea_level_equivalent(time_s: float, altitude_m: int, event: str) -> float:
    """Return estimated sea-level equivalent performance (seconds)."""
    coeff = ALTITUDE_CORRECTIONS.get(event, 0.0)
    correction = coeff * (altitude_m / 1000.0)
    return time_s - correction
```

- Returns original `time_s` unchanged when event not in map (safe default).
- Piecewise corrections can be added later without changing the function signature.

---

## Phase 3 — Preprocessor Update

**File: `app/core/preprocessor.py`**

After joining altitude data, add three derived columns:

| Column | Formula | Purpose |
|--------|---------|---------|
| `altitude_band` | `pd.cut(altitude_m, bins=[…])` | Categorical feature for ML |
| `altitude_correction_s` | `sea_level_equivalent(time, alt, event) - time` | Raw adjustment amount |
| `Time_sea_level` | `Time + altitude_correction_s` | Sea-level equivalent time |

Use `Time_sea_level` in all downstream computation where available; fall back to `Time`
where `altitude_m` is 0 (default_sea_level rows).

---

## Phase 4 — Statistical Detectors

**Files:** `app/detection/statistical/zscore.py`, `mad.py`, `iqr.py`

Swap the reference series from `Time` to `Time_sea_level`:

```python
# Before
series = df["Time"]
# After
series = df["Time_sea_level"].fillna(df["Time"])
```

This ensures altitude-boosted performances (e.g., high-altitude sprints) are not flagged
as anomalies purely due to altitude benefit.

---

## Phase 5 — ML Detectors

**Files:** `app/detection/ml/isolation_forest.py`, `xgboost_detector.py`, `graph.py`

Add altitude features to feature matrices:

```python
altitude_features = [
    "altitude_m",             # raw altitude
    "altitude_band_enc",      # encoded categorical band
    "altitude_correction_s",  # magnitude of correction
]
```

Use `Time_sea_level` as the primary performance input (not raw `Time`).

For the graph detector, `altitude_band` can be a node attribute enabling altitude-aware
edge weighting between performances.

---

## Phase 6 — Ablation Evaluation

Run the detector suite twice:

1. **Baseline** — original pipeline without altitude features (current state)
2. **Altitude-corrected** — pipeline with `Time_sea_level` + altitude features

Compare:
- False positive rate at high-altitude venues (Mexico City, Nairobi, Addis Ababa)
- Precision/recall on known doping positives (if a labelled test set exists)
- Agreement between statistical and ML detectors

Report the delta to quantify the benefit of the altitude integration.

---

## Verification Checklist

- [ ] `sql_audit.py` shows 0 unmatched venues (or only the 6 known default_sea_level rows)
- [ ] DuckDB view returns altitude columns for a sample query against `crawled_competitions`
- [ ] `data_loader.py` DataFrame has `altitude_m`, `altitude_source`, `Time_sea_level` columns
- [ ] Sea-level correction for Mexico City 100m (2,240 m): ~0.067s adjustment (negligible for sprints — confirms correct sign)
- [ ] Sea-level correction for Nairobi 800m (1,795 m): ~−0.25s (time *increases* for aerobic events — correct sign)
- [ ] Statistical detectors produce fewer false positives for Nairobi/Addis Ababa middle-distance results
- [ ] Ablation comparison documented in research report

---

## Key File Paths

| File | Location | Status |
|------|----------|--------|
| `venue_altitudes` table | PostgreSQL (see `.env`) | ✅ Complete |
| `bulk_inference.py` | `altitude-research/bulk_inference.py` | ✅ Complete |
| `sql_audit.py` | `altitude-research/sql_audit.py` | ✅ Complete |
| `duckdb_client.py` | `app/db/duckdb_client.py` | 🔲 Phase 1.1 |
| `data_loader.py` | `app/core/data_loader.py` | 🔲 Phase 1.2 |
| `altitude.py` | `app/core/altitude.py` (new file) | 🔲 Phase 2 |
| `preprocessor.py` | `app/core/preprocessor.py` | 🔲 Phase 3 |
| Statistical detectors | `app/detection/statistical/` | 🔲 Phase 4 |
| ML detectors | `app/detection/ml/` | 🔲 Phase 5 |

---

## Important Notes

- The `OR c.venue = va.raw_venue_string` join condition **must** be used everywhere —
  without it, the 135 orphaned raw strings silently return NULL altitude.
- The 6 default_sea_level rows have `altitude_m = 0` — their correction is 0s, which is
  the correct conservative behaviour for unresolvable venues.
- Re-running `rerun_remaining.py` or `rerun_low_confidence.py` is safe (idempotent) if
  more venues are added to `crawled_competitions` in future.
