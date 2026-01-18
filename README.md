# Venue Altitude Inference with LLMs

**Inferring stadium/venue altitudes from competition mentions using frontier LLMs.**  
Zero-shot and few-shot prompting experiments with GPT-5.2, Claude 4.5 Sonnet, Gemini 2.5 Pro, and Qwen 3-8B. Real OpenRouter API costs tracked. Evaluated on 145 gold-standard venues.

## Project Overview

This repository demonstrates using large language models to extract venue elevation (meters above sea level) from unstructured sports competition mentions (e.g., "Olympic Stadium, Helsinki (FIN) – World Championships – 2022-08-19").

### Key Features
- Four prompt variants (short, recent mentions, old+new, few-shot)
- Concurrent LLM calls via `asyncio` + `litellm`
- Gold-standard evaluation (145 hand-verified venues)
- Real per-call cost tracking from OpenRouter exports
- PostgreSQL + DuckDB hybrid storage
- Geocoding + SRTM/ASTER elevation baselines

## Repository Structure

| File / Folder                        | Purpose                                                                 |
|--------------------------------------|-------------------------------------------------------------------------|
| `main.py`                            | Main pipeline: loads gold venues, runs concurrent LLM queries, saves predictions to DB + CSV backup |
| `prompts.py`                         | Four prompt templates (v1–v4): short, recent mentions, old+new, few-shot |
| `llm.py`                             | LLM wrapper classes (ChatGPT, Gemini, Claude, Qwen) using `litellm` + OpenRouter |
| `gold_standard_venue.py`             | Builds gold dataset: geocodes venues, fetches SRTM/ASTER elevations, selects best value |
| `get_db_venues.py`                   | Queries PostgreSQL → extracts unique venues + mentions → exports to CSV |
| `db.py`                              | Database initialization (Postgres schema + DuckDB → Postgres bridge) + store helper |
| `analysis.py`                        | Loads predictions + OpenRouter CSV costs → computes MAE, % large errors, unknown rate, real costs → ranks models/prompts |
| `data/`                              | Gold venues, distinct competitions, OpenRouter activity CSVs, predictions backup |
| `.env.example`                       | Template for API keys and DB credentials (copy → `.env`) |

## How the Application Flows

1. **Data Preparation**  
   `get_db_venues.py` → extracts ~4,250 unique venues + mentions from `crawled_competitions` table → saves `distinct_venue_comps.csv`

2. **Gold Standard Creation**  
   `gold_standard_venue.py` → geocodes venues (Google), fetches SRTM/ASTER elevations (OpenTopography), selects best value → saves `gold_venues_with_elevation.csv` + stores in `venues_gold_standard` table

3. **LLM Inference**  
   `main.py` → loads gold venues → for each venue:
   - generates 4 prompt variants
   - queries 4 LLMs concurrently (`asyncio.gather`)
   - collects structured predictions
   - saves to `venues_predictions` table + CSV backup

4. **Analysis & Cost Tracking**  
   `analysis.py` → loads predictions + OpenRouter CSV → matches via model + ordered timestamps → computes:
   - MAE, median AE, % errors >100m
   - % unknown answers
   - real per-query and total costs
   → ranks models/prompts by combined accuracy + cost score

## How to Run

### Prerequisites
- Python 3.10+
- PostgreSQL database (local or cloud)
- OpenRouter account (free tier ok for small runs)

### Setup
1. Clone the repo
   ```bash
   git clone https://github.com/blessedmadukoma/llm-venue-altitude-extraction-research.git venue-altitude-llm
   cd venue-altitude-llm

2. Create & activate virtual environmentBashpython -m venv .venv
    ```bash
    source .venv/bin/activate          # Linux/macOS
    # or .venv\Scripts\activate        # Windows
    ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Copy and fill .env with your keys
   ```Bash
   cp .env.example .env
   ```

   Edit .env with your keys:
   ```text
   OPENROUTER_API_KEY=sk-or-v1-...
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=your_db
   DB_USER=your_user
   DB_PASSWORD=your_pass
   GOOGLE_GEOCODING_API_KEY=AIza...
   ```
5. Initialize database (creates tables)
   ```Bash
   python db.py
   ```

6. Run the pipeline:
   1. Extract venues from DB → data/distinct_venue_comps.csv
    ```Bash
    python get_db_venues.py
    ```
   2. Build gold standard (geocode + elevation) → data/gold_venues_with_elevation.csv
   ```Bash
   python gold_standard_venue.py
   ```
   3. Run LLM inference on gold venues → saves to DB + data/llm_venues_predictions.csv
   ```Bash
   python main.py
   ```
    4. Analyze results + real costs (download your cost from OpenRouter)
   ```Bash
   python analysis.py
   ```

### **Notes:**
- **Costs:** Tracked via OpenRouter dashboard exports — real billed USD, no estimates.
- **Concurrency:** Limited to 3 venues at once (adjust Semaphore(3) in main.py if needed).
- **Reproducibility:** Use same .env keys and data snapshot for exact replication.