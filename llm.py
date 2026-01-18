from abc import ABC, abstractmethod
from typing import Dict
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
                "altitude_meters": "Unknown",
                "confidence": "Low",
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
