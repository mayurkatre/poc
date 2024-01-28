"""
OpenRouter Client Utility
==========================
Provides a pre-configured OpenAI client pointed at OpenRouter's API.

OpenRouter is a unified API gateway that supports 200+ LLMs from providers
like OpenAI, Anthropic, Meta, Google, and Mistral — all through the standard
OpenAI SDK interface.

Usage:
    from config.openrouter import get_client

    client = get_client()
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )

Environment variables required (set in .env):
    OPENAI_API_KEY      — Your OpenRouter key (starts with sk-or-v1-)
    OPENAI_BASE_URL     — https://openrouter.ai/api/v1

Optional (shown in OpenRouter usage dashboard):
    OPENROUTER_APP_URL   — Your app's URL or GitHub repo
    OPENROUTER_APP_TITLE — Your app's display name
"""

from __future__ import annotations

import os
from functools import lru_cache

import openai
from loguru import logger


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@lru_cache(maxsize=1)
def get_client() -> openai.OpenAI:
    """
    Create and cache a single OpenRouter-configured OpenAI client.

    Reads configuration from environment variables. The client is cached
    so only one instance is created per process.

    Returns:
        openai.OpenAI: Client configured for OpenRouter.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Get your OpenRouter key at https://openrouter.ai/keys "
            "and add it to your .env file."
        )

    base_url = os.getenv("OPENAI_BASE_URL", OPENROUTER_BASE_URL)

    # Optional headers shown in OpenRouter dashboard
    app_url = os.getenv("OPENROUTER_APP_URL", "https://github.com/your-username/rag-poc")
    app_title = os.getenv("OPENROUTER_APP_TITLE", "RAG POC")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": app_url,
            "X-Title": app_title,
        },
    )

    logger.debug(f"OpenRouter client initialized (base_url={base_url})")
    return client


def get_streaming_client() -> openai.OpenAI:
    """
    Get the OpenRouter client for streaming responses.

    Same as get_client() — OpenRouter streaming uses the same client,
    just with stream=True in the API call.

    Returns:
        openai.OpenAI: Client configured for OpenRouter.
    """
    return get_client()
