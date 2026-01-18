import asyncio
import warnings
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

import litellm
from db import init_db, store_to_table
from get_db_venues import pretty_table, distinct_venue_comps
from gold_standard_venue import select_gold_standard
from llm import ChatGPT, Gemini, Claude, Qwen
import time
from prompts import (
    PROMPT_1_TEMPLATE_SHORT,
    PROMPT_2_TEMPLATE_RECENT,
    PROMPT_3_TEMPLATE_OLD_NEW,
    PROMPT_4_TEMPLATE_FEW_SHOT,
)

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module="litellm.*")


def prepare_prompt_variants(venue_row) -> List[Dict[str, str]]:
    """Generate all prompt variants for one venue"""
    canonical = venue_row.canonical_venue

    recent = "\n".join(
        venue_row.recent_mentions[:6]) if venue_row.recent_mentions else ""
    oldest = "\n".join(
        venue_row.oldest_mentions[:6]) if venue_row.oldest_mentions else ""
    newest = "\n".join(
        venue_row.newest_mentions[:6]) if venue_row.newest_mentions else ""

    variants = [
        ("prompt_v1", PROMPT_1_TEMPLATE_SHORT.format(canonical_venue=canonical)),
        ("prompt_v2", PROMPT_2_TEMPLATE_RECENT.format(
            canonical_venue=canonical, recent_mentions=recent or "N/A")),
        ("prompt_v3", PROMPT_3_TEMPLATE_OLD_NEW.format(canonical_venue=canonical,
         oldest_mentions=oldest or "N/A", newest_mentions=newest or "N/A")),
        ("prompt_v4", PROMPT_4_TEMPLATE_FEW_SHOT.format(
            canonical_venue=canonical, recent_mentions=recent or "N/A")),
    ]
    return [{"version": v, "text": p} for v, p in variants]


async def query_llm_with_timeout(llm, prompt: str, timeout: float = 120.0) -> Dict[str, Any]:
    """Single LLM call with timeout"""
    try:
        result = await asyncio.wait_for(llm.query(prompt), timeout=timeout)
        return result if isinstance(result, dict) else {"error": "Unexpected result format"}
    except asyncio.TimeoutError:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}


async def process_one_venue(venue_row, llms: List) -> List[Dict]:
    """Process one venue → returns list of prediction dicts"""
    canonical = venue_row.canonical_venue
    print(f"→ Processing venue: {canonical}")

    prompt_variants = prepare_prompt_variants(venue_row)
    predictions = []

    for prompt_info in prompt_variants:
        version = prompt_info["version"]
        prompt = prompt_info["text"]

        # Launch all LLM calls for this prompt concurrently
        tasks = []
        for llm in llms:
            model_name = getattr(llm, "get_model", lambda: "unknown")()
            print(f"  ↳ {model_name}  ({version})", end=" ... ", flush=True)
            tasks.append(
                asyncio.create_task(
                    query_llm_with_timeout(llm, prompt)
                )
            )

        # Wait for all models to finish this prompt
        results = await asyncio.gather(*tasks, return_exceptions=True)

        now = datetime.utcnow()

        for llm, result in zip(llms, results):
            model_name = getattr(llm, "get_model", lambda: "unknown")()

            if isinstance(result, Exception):
                print(f"FAILED ({str(result)[:60]})")
                continue

            if "error" in result:
                print(f"FAILED ({result['error'][:60]})")
                continue

            altitude = result.get("altitude_meters", "Unknown")
            conf = str(result.get("confidence", "low")).lower()
            reasoning = result.get("source", "")

            row = {
                "canonical_venue": canonical,
                "model_family": model_name.split("/")[-1].split(":")[0],
                "model_version": model_name,
                "prompt_version": version,
                "predicted_altitude": altitude,
                "confidence_from_model": conf,
                "reasoning_text": reasoning,
                "run_timestamp": now,
                "run_id": f"{canonical}_{version}_{model_name}_{now.isoformat()}",
            }

            predictions.append(row)
            print(f"OK → {altitude}  ({conf})")

    print("─" * 70)
    return predictions


async def main():
    conn = None
    try:
        conn = init_db()
        print("DB initialized")

        # Gold standard
        print("Loading gold standard venues...")
        # gold_std_venues = select_gold_standard("./data/distinct_venue_comps.csv")
        gold_std_venues = pd.read_csv("./data/gold_venues_with_elevation.csv")
        print(f"→ {len(gold_std_venues)} venues loaded\n")

        # models
        llms = [
            ChatGPT(model="gpt-5.2"),
            Gemini(model="gemini-2.5-pro"),
            Claude(),
            Qwen(),

            # Base and cheap options
            # ChatGPT(),
            # Gemini(),
            # Claude(),
            # Qwen(),
            # Ollama(),
        ]

        print(f"Using {len(llms)} models\n")

        all_predictions = []

        # Concurrency: process 3 venues at once
        semaphore = asyncio.Semaphore(3)

        async def sem_process(venue):
            async with semaphore:
                return await process_one_venue(venue, llms)

        tasks = [sem_process(row)
                 for row in gold_std_venues.itertuples(index=False)]
        venue_results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in venue_results:
            if isinstance(res, Exception):
                print("Venue failed:", res)
                continue
            all_predictions.extend(res)

        if all_predictions:
            df = pd.DataFrame(all_predictions)
            print(f"\nCollected {len(df)} predictions")

            cols = [
                "canonical_venue", "model_family", "model_version", "prompt_version",
                "predicted_altitude", "confidence_from_model", "reasoning_text",
                "run_timestamp", "run_id"
            ]

            df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])

            # backup
            df.to_csv("./data/llm_venues_predictions.csv")

            store_to_table("venues_predictions", cols, df)
            print("→ Saved to database")

    except Exception as e:
        print("Main loop failed:", e)

    finally:
        if conn is not None:
            try:
                await litellm.close_litellm_async_clients()
            except:
                pass
            try:
                conn.close()
                print("DB connection closed")
            except:
                pass


if __name__ == "__main__":
    _start = time.perf_counter()
    asyncio.run(main())
    _elapsed = time.perf_counter() - _start

    _hours, _rem = divmod(_elapsed, 3600)
    _minutes, _seconds = divmod(_rem, 60)
    print(
        f"Total runtime: {int(_hours)}h {int(_minutes)}m {_seconds:.2f}s ({_elapsed:.2f} sec)")
