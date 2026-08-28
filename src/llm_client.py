"""
LLM client abstraction.

Provides:
1. MockLLMClient for deterministic offline testing.
2. OllamaClient for real local-model inference.
"""

import json
from abc import ABC, abstractmethod

import requests


class LLMClient(ABC):
    """Base interface for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """
    Deterministic mock client.

    Used for testing the pipeline without requiring
    a real LLM.
    """

    def generate(self, prompt: str) -> str:
        """
        Return a deterministic JSON response.

        This is ONLY for pipeline testing.
        It must not be used for the final benchmark.
        """

        return json.dumps({
            "results": []
        })


class OllamaClient(LLMClient):
    """
    Client for a locally running Ollama model.
    """

    def __init__(
        self,
        model="qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        timeout=300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the generated text.
        """

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": 1200,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]