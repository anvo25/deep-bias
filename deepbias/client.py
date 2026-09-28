"""Thin OpenAI-compatible client used by the closed-models eval pipeline.

Only `TargetClient` is exposed here; the closed-models pipeline never
calls a mutator/judge.  Copied from `legacy/src_core_qd/core/clients.py`
during the 2026-05 pipeline-repair pass.

`TargetClient` talks the OpenAI chat.completions API.  It works against:
  - vLLM-served local models (`base_url=http://localhost:PORT/v1`)
  - OpenAI / OpenRouter / Cohere / Anthropic / Google AI Studio OpenAI-compat endpoints

It handles a few provider quirks transparently:
  - Cohere + Anthropic OpenAI-compat reject `n>1`; we emulate n>1 by N
    sequential calls.
  - GPT-5 / o-series reasoning models accept `max_completion_tokens` and
    reject `max_tokens`; we switch the key based on the model id.
  - OpenRouter Gemini 3.x routes force reasoning ON; we disable it via
    extra_body {"reasoning": ...} per-model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

import os

# Appended to every direct and framed prompt so the model emits a bare answer.
ANSWER_SUFFIX = (
    "Pick exactly ONE option. Respond with only your choice, "
    "1-3 words, no other text."
)

# Which environment variable holds the key for which endpoint. Local vLLM
# servers need no real key.
_KEY_ENV = [
    ("openai.com", "OPENAI_API_KEY"),
    ("openrouter.ai", "OPENROUTER_API_KEY"),
    ("anthropic.com", "ANTHROPIC_API_KEY"),
]


def resolve_api_key(base_url: str, explicit: str | None = None) -> str:
    """Return the API key for `base_url`, read from the environment."""
    if explicit:
        return explicit
    for host, env_var in _KEY_ENV:
        if host in (base_url or ""):
            key = os.environ.get(env_var)
            if not key:
                raise SystemExit(f"{base_url} needs an API key: set ${env_var}.")
            return key
    return "EMPTY"  # a local vLLM server accepts any key


def _make_http_client(timeout: float) -> httpx.Client:
    """Connection pool sized for high-concurrency fan-out; the default
    openai-SDK pool caps around 100 connections, which silently bottlenecks
    requests when concurrency is in the hundreds."""
    return httpx.Client(
        limits=httpx.Limits(
            max_connections=1024, max_keepalive_connections=512,
        ),
        timeout=timeout,
    )


@dataclass
class TargetClient:
    """Samples N completions per call from an OpenAI-compatible endpoint."""
    model: str
    base_url: str
    api_key: str = "x"
    timeout: float = 60.0
    reasoning_effort: str | None = None
    disable_thinking: bool = False

    def __post_init__(self):
        self._client = OpenAI(
            base_url=self.base_url, api_key=self.api_key,
            timeout=self.timeout, http_client=_make_http_client(self.timeout),
        )

    def sample(
        self,
        prompt: str,
        n: int,
        temperature: float = 1.0,
        max_tokens: int = 25,
        guided_regex: str | None = None,
        guided_choice: list[str] | None = None,
        system: str | None = None,
        reasoning_effort: str | None = None,
        disable_thinking: bool | None = None,
    ) -> list[str]:
        """Return n completion strings (stripped)."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        extra_body: dict[str, Any] = {}
        if guided_regex:
            extra_body["guided_regex"] = guided_regex
        if guided_choice:
            extra_body["guided_choice"] = guided_choice
        eff = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        if eff is not None:
            extra_body["reasoning_effort"] = eff
        dt = disable_thinking if disable_thinking is not None else self.disable_thinking
        if dt:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        # OpenRouter expects `reasoning.effort` nested, not a flat
        # `reasoning_effort` field.  Translate so callers can keep using the
        # OpenAI-native param shape.
        if "openrouter.ai" in (self.base_url or "") and "reasoning" not in extra_body:
            if "reasoning_effort" in extra_body:
                extra_body["reasoning"] = {"effort": extra_body.pop("reasoning_effort")}
            else:
                FORCED_REASONING_MODELS = (
                    "google/gemini-3.5-flash",
                    "google/gemini-3.5-pro",
                    "google/gemini-3-flash-preview",
                    "google/gemini-3.1-pro-preview",
                )
                if any(self.model.startswith(m) for m in FORCED_REASONING_MODELS):
                    extra_body["reasoning"] = {"max_tokens": 10}
                else:
                    extra_body["reasoning"] = {"enabled": False}

        if "generativelanguage.googleapis.com" in (self.base_url or ""):
            extra_body.pop("reasoning", None)

        # Normalize provider-prefixed model ids ("openai/gpt-5.1") so the
        # GPT-5 / o-series detector still fires on OpenRouter routes.
        _model_base = self.model.split("/")[-1] if "/" in self.model else self.model
        is_gpt5_reasoning = _model_base.startswith("gpt-5") or _model_base.startswith("o")
        is_n1_only_compat = (
            "cohere.com" in (self.base_url or "")
            or "api.anthropic.com" in (self.base_url or "")
        )

        kwargs: dict[str, Any] = dict(
            model=self.model, messages=messages,
            extra_body=extra_body or None,
        )
        if not is_n1_only_compat:
            kwargs["n"] = n
        if is_gpt5_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        if is_n1_only_compat and n > 1:
            outs: list[str] = []
            for _ in range(n):
                resp = self._client.chat.completions.create(**kwargs)
                outs.append((resp.choices[0].message.content or "").strip()
                            if resp.choices else "")
            return outs

        resp = self._client.chat.completions.create(**kwargs)
        return [(c.message.content or "").strip() for c in resp.choices]

    def sample_completion(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.6,
        max_tokens: int = 15,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Return n raw-completion strings (stripped) via the legacy
        /v1/completions endpoint, no chat template applied.

        Use this instead of sample() for base/pretrained models: they were
        never instruction-tuned, so sending them a chat-formatted "answer
        in one word" instruction through /v1/chat/completions is
        unreliable (some have no chat template at all, others just
        ramble). A plain completion prompt (e.g. "Q: ...\\nA:") with a
        newline stop sequence is the format they actually saw in
        pretraining. See legacy/src_curation_unused/basemodel_inference/
        eval_olmo_compare.py for the original transformers-based version
        of this same idea.
        """
        resp = self._client.completions.create(
            model=self.model, prompt=prompt, n=n,
            temperature=temperature, max_tokens=max_tokens,
            stop=stop,
        )
        return [(c.text or "").strip() for c in resp.choices]
