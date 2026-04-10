from abc import ABC, abstractmethod
from typing import Dict
import asyncio
import litellm
import json
from time import time
import os


class BaseLLM(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def query(self, prompt: str) -> Dict:
        """Send prompt to LLM and return structured response"""
        pass

    def get_model(self) -> str:
        return self.model


class ResponseLLM(BaseLLM):
    """
    Base class for OpenAI / Gemini / Claude-style chat models.
    """

    def __init__(self, model: str, api_base: str | None = None, api_key: str | None = None):
        super().__init__(model=model)
        self.api_base = api_base
        self.api_key = api_key

    async def query(self, prompt: str) -> Dict:
        start = time()

        response = await litellm.acompletion(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        duration = time() - start
        print(f"{self.model} response time: {duration:.2f}s")

        content = response.choices[0].message.content.strip()

        if content.startswith("```json"):
            content = content.split("```json", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        try:
            parsed = json.loads(content)

            # validate keys
            required_keys = {"canonical_venue",
                             "altitude_meters", "confidence", "source"}
            if not required_keys.issubset(parsed.keys()):
                raise ValueError("Missing required keys")

            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parse error for {self.model}: {e}")
            print("Raw content:", content)
            return {
                "canonical_venue": "ERROR",
                "altitude_meters": -1,
                "confidence": "low",
                "source": f"Parse failed: {str(e)}"
            }


class ChatGPT(ResponseLLM):
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        super().__init__(
            model=f"openai/{model}",
            api_base=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )


class Gemini(ResponseLLM):
    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(
            model=f"openrouter/google/{model}",
            api_base=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )


class Claude(ResponseLLM):
    def __init__(self, model: str = "claude-4.5-sonnet"):
        super().__init__(
            model=f"openrouter/anthropic/{model}",
            api_base=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )


class Qwen(ResponseLLM):
    def __init__(self, model: str = "qwen3-8b"):
        super().__init__(
            model=f"openrouter/qwen/{model}",
            api_base=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )


class Ollama(ResponseLLM):
    # Not in use
    def __init__(self, model: str = "mistral:latest"):
        super().__init__(
            model=f"openrouter/qwen/{model}",
            api_base=os.getenv("OLLAMA_BASE_URL"),
        )


class BulkGemini(ResponseLLM):
    """
    Gemini variant for bulk venue inference (PROMPT_BULK_TEMPLATE).

    Primary:  Google AI Studio (GEMINI_API_KEY) — direct, no middleman cost.
    Fallback: OpenRouter (OPENROUTER_API_KEY) — used if direct call fails.

    Validates the bulk response schema:
      raw_venue_string, canonical_venue, altitude_meters, confidence, reasoning
    altitude_meters is always coerced to an integer; -1 means undetermined.
    """

    _REQUIRED_KEYS = {"raw_venue_string", "canonical_venue",
                      "altitude_meters", "confidence", "reasoning"}

    # def __init__(self, model: str = "gemini-2.5-pro"):
    def __init__(self, model: str = "gemini-3-flash-preview"):
        self._model_slug = model

        # Primary: Google AI Studio direct
        self._direct_model = f"gemini/{model}"
        self._direct_key = os.getenv("GEMINI_API_KEY")

        # Fallback: OpenRouter
        self._fallback_model = f"openrouter/google/{model}"
        self._fallback_base = os.getenv("OPENROUTER_BASE_URL")
        self._fallback_key = os.getenv("OPENROUTER_API_KEY")

        # Parent stores self.model (used by get_model()); point it at the direct path
        super().__init__(model=self._direct_model, api_key=self._direct_key)

    def _parse(self, content: str) -> Dict:
        """Strip markdown fences, parse JSON, validate keys, coerce types."""
        if content.startswith("```json"):
            content = content.split("```json", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        parsed = json.loads(content)

        missing = self._REQUIRED_KEYS - parsed.keys()
        if missing:
            raise ValueError(f"Missing keys: {missing}")

        alt = parsed["altitude_meters"]
        if isinstance(alt, str):
            alt = int(alt) if alt.lstrip("-").isdigit() else -1
        elif isinstance(alt, float):
            alt = int(alt)
        parsed["altitude_meters"] = alt
        parsed["confidence"] = str(parsed.get("confidence", "low")).lower()
        return parsed

    # Retry config for rate-limit errors on the direct API
    _MAX_RETRIES = 4
    _BACKOFF_BASE = 5   # seconds: waits 5, 10, 20, 40 s between retries

    async def query(self, prompt: str) -> Dict:
        start = time()
        raw_venue = (
            prompt.split("Venue to process:")[-1].split("\n")[0].strip()
            if "Venue to process:" in prompt else "ERROR"
        )

        # ── 1. Google AI Studio (primary) — retry on rate limit ──────────────
        if self._direct_key:
            for attempt in range(self._MAX_RETRIES):
                try:
                    resp = await litellm.acompletion(
                        model=self._direct_model,
                        api_key=self._direct_key,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                    )
                    parsed = self._parse(
                        resp.choices[0].message.content.strip())
                    print(
                        f"gemini-direct {time()-start:.2f}s", end=" ", flush=True)
                    return parsed

                except litellm.RateLimitError:
                    wait = self._BACKOFF_BASE * \
                        (2 ** attempt)   # 5, 10, 20, 40 s
                    if attempt < self._MAX_RETRIES - 1:
                        print(
                            f"\n  [rate limit — retry {attempt+1}/{self._MAX_RETRIES} in {wait}s]", flush=True)
                        await asyncio.sleep(wait)
                    else:
                        print(
                            f"\n  [rate limit — all retries exhausted → OpenRouter]", flush=True)

                except Exception as e:
                    print(
                        f"\n  [direct error: {str(e)[:70]} → OpenRouter]", flush=True)
                    break   # non-rate-limit error: skip retries, fall through now

        # ── 2. OpenRouter (fallback) ─────────────────────────────────────────
        try:
            resp = await litellm.acompletion(
                model=self._fallback_model,
                api_base=self._fallback_base,
                api_key=self._fallback_key,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            parsed = self._parse(resp.choices[0].message.content.strip())
            print(f"openrouter {time()-start:.2f}s", end=" ", flush=True)
            return parsed
        except Exception as e:
            print(f"All APIs failed: {e}")
            return {
                "raw_venue_string": raw_venue,
                "canonical_venue":  raw_venue,
                "altitude_meters": -1,
                "confidence":       "low",
                "reasoning":        f"Both APIs failed: {str(e)}",
            }
