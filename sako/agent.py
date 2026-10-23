"""Bounded Gemini requests. Credentials stay outside project files."""

import json
import os
import threading
import time
from pathlib import Path

from .core import write_json


class Gemini:
    def __init__(self, config, store, run_dir):
        self.config, self.store, self.run_dir = config, store, Path(run_dir)
        self.gate = threading.BoundedSemaphore(config.get("max_parallel_calls", 2))
        self.cooldown_lock = threading.Lock()
        self.not_before = 0.

    def _request(self, prompt, system):
        from google import genai
        from google.genai import types
        config = self.config
        client = genai.Client(vertexai=True, project=config["project"], location=config["location"],
            http_options=types.HttpOptions(timeout=config["timeout_seconds"] * 1000,
                                           retry_options=types.HttpRetryOptions(attempts=1)))
        try:
            response = client.models.generate_content(model=config["model"], contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system,
                    response_mime_type="application/json", temperature=.7,
                    max_output_tokens=config["max_output_tokens"],
                    thinking_config=types.ThinkingConfig(thinking_level="LOW")))
            return response.text or "", response.usage_metadata.total_token_count if response.usage_metadata else None
        finally:
            client.close()

    def call(self, role, prompt, system):
        config = self.config
        if len(prompt) > config["context_chars"]:
            raise ValueError("Prompt exceeds the context budget. Remove old observations first.")
        for attempt in range(config.get("max_retries", 2) + 1):
            try:
                with self.gate:
                    with self.cooldown_lock:
                        delay = max(0., self.not_before - time.monotonic())
                    if delay:
                        time.sleep(delay)
                    return self._attempt(role, prompt, system)
            except Exception as error:
                code = getattr(error, "code", None)
                if code not in (429, 500, 502, 503, 504) or attempt >= config.get("max_retries", 2):
                    raise
                with self.cooldown_lock:
                    self.not_before = max(self.not_before, time.monotonic() + 30 * (2 ** attempt))

    def _attempt(self, role, prompt, system):
        config = self.config
        # UTF-8 bytes form a conservative token reserve. Output includes thinking tokens.
        reserve = len((prompt + system).encode()) + config["max_output_tokens"]
        call_id = self.store.reserve(role, reserve, config)
        path = self.run_dir / "calls" / f"{call_id:05d}.json"
        write_json(path, {"role": role, "system": system, "prompt": prompt, "status": "pending"})
        tokens = None
        try:
            text, tokens = self._request(prompt, system)
            self.store.finish_call(call_id, tokens)
            write_json(path, {"role": role, "system": system, "prompt": prompt, "response": text,
                              "tokens": tokens, "model": config["model"], "status": "complete"})
            return json.loads(text)
        except Exception as error:
            # Keep the reserve for an unknown outcome. This prevents a retry from hiding cost.
            self.store.finish_call(call_id, 0 if getattr(error, "code", None) == 429 else tokens, status="failed")
            write_json(path, {"role": role, "system": system, "prompt": prompt,
                              "error": f"{type(error).__name__}: {error}", "status": "failed"})
            raise
