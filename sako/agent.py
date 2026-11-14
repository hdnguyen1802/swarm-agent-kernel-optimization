"""Shared request budgets, pacing, and provider selection."""

import json
import random
import threading
import time
import traceback
from pathlib import Path

from .core import write_json, digest
from .transport import TransferError


class AuthenticationUnavailable(TransferError):
    """Credentials failed before dispatch to a generation endpoint."""
    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = retryable


class AdaptiveGate:
    """Reduce new concurrency after throttling; recover within the fixed ceiling."""
    def __init__(self, maximum, effective=None):
        self.maximum = maximum
        self.limit = min(maximum, max(1, effective or maximum))
        self.active = 0
        self.successes = 0
        self.condition = threading.Condition()

    def __enter__(self):
        with self.condition:
            self.condition.wait_for(lambda: self.active < self.limit)
            self.active += 1
        return self

    def __exit__(self, *_):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def throttle(self):
        with self.condition:
            self.limit = max(1, self.limit // 2)
            self.successes = 0
            return self.limit

    def succeed(self):
        with self.condition:
            self.successes += 1
            if self.limit < self.maximum and self.successes >= self.maximum:
                self.limit += 1
                self.successes = 0
                self.condition.notify_all()
            return self.limit


def validate_provider(config):
    provider = config.get("provider", "vertex")
    if not isinstance(provider, str) or provider not in {"vertex", "deepseek"}:
        raise ValueError("Use vertex or deepseek for agent.provider.")
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ValueError("Set the agent model name for the selected provider.")
    if "api_key" in config:
        raise ValueError("Keep API keys outside configs. Use an environment variable.")
    if provider == "vertex":
        if any(not isinstance(config.get(k), str) or not config[k].strip() for k in ("project", "location")):
            raise ValueError("Vertex requires agent.project and agent.location.")
    else:
        import re
        variable = config.get("api_key_env", "DEEPSEEK_API_KEY")
        if not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
            raise ValueError("Use an environment variable name for agent.api_key_env.")
        thinking = config.get("thinking", "enabled")
        if not isinstance(thinking, str) or thinking not in {"enabled", "disabled"}:
            raise ValueError("Use enabled or disabled for agent.thinking.")
        effort = config.get("reasoning_effort", "low")
        if not isinstance(effort, str) or effort not in {"low", "high", "max"}:
            raise ValueError("Use low, high, or max for agent.reasoning_effort.")
        if any(type(config.get(k)) is not int or config[k] <= 0
               for k in ("model_context_tokens", "model_output_tokens")):
            raise ValueError("DeepSeek requires verified model_context_tokens and model_output_tokens.")
        if (type(config.get("max_output_tokens")) is not int or config["max_output_tokens"] <= 0
                or config["max_output_tokens"] > config["model_output_tokens"]):
            raise ValueError("The requested output exceeds the configured DeepSeek output limit.")


def create_agent(config, store, run_dir):
    validate_provider(config)
    if config.get("provider", "vertex") == "deepseek":
        from .deepseek import DeepSeek
        return DeepSeek(config, store, run_dir)
    return Gemini(config, store, run_dir)


class Agent:
    def __init__(self, config, store, run_dir):
        self.config, self.store, self.run_dir = config, store, Path(run_dir)
        pacing = self.store.get("api_pacing", {})
        self.gate = AdaptiveGate(config.get("max_parallel_calls", 4), pacing.get("effective_parallel_calls"))
        self.cooldown_lock = threading.Lock()
        self.not_before = time.monotonic() + max(0., pacing.get("not_before_utc", 0.) - time.time())
        self.provider_limits = None
        self.provider_limits_lock = threading.Lock()
        self.request_locks = {}
        self.request_locks_guard = threading.Lock()

    def _token_count_metadata(self):
        return {}

    def _cache_settings(self):
        return {}

    def _check_response(self, text, usage):
        pass

    def _auxiliary(self, role, endpoint, request, action):
        """Record metadata traffic separately from generation attempts."""
        call_id = self.store.reserve(role + ":" + endpoint, 0, self.config)
        path = self.run_dir / "calls" / f"{call_id:05d}.json"
        record = {"role": role, "endpoint": endpoint, "request": request, "status": "pending",
                  "tokens": 0, "token_accounting": "Metadata request; separate from generation attempts."}
        write_json(path, record)
        try:
            result = action()
            record.update(status="complete", response=result)
            self.store.finish_call(call_id, 0)
            write_json(path, record)
            return result
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            self.store.finish_call(call_id, 0, status="failed")
            write_json(path, record)
            raise

    def limits_for_role(self, role, prompt=None):
        limits = {**self.config}
        limits.update(self.config.get("role_limits", {}).get(role, {}))
        if role == "coordinator" and self.config.get("search_mode") == "swarm":
            from .swarm_policy import coordinator_response_schema
            limits["max_input_tokens"] = self.config.get("coordinator_max_input_tokens", 150000)
            actions = ["continue", "move_on", "wait"]
            try:
                context = json.loads(prompt) if prompt else {}
            except json.JSONDecodeError:
                context = {}
            if isinstance(context, dict) and "allowed_actions" in context:
                allowed = context["allowed_actions"]
                if not isinstance(allowed, list) or not allowed or any(a not in actions for a in allowed):
                    raise ValueError("The plan request must specify valid coordinator actions.")
                actions = allowed
            limits["response_schema"] = coordinator_response_schema(actions)
        elif self.config.get("adaptive_budget"):
            limits.setdefault("max_input_tokens", self.config.get("model_context_tokens", 0))
        return limits

    def _check_token_limit(self, role, prompt, system, limits):
        with self.provider_limits_lock:
            if self.provider_limits is None:
                configured_context = self.config.get("model_context_tokens")
                configured_output = self.config.get("model_output_tokens")
                if configured_context is not None:
                    self.provider_limits = {"input_token_limit": configured_context,
                                            "output_token_limit": configured_output,
                                            "source": "explicit_config"}
                else:
                    self.provider_limits = self._auxiliary(role, "models.get",
                        {"model": self.config["model"]}, self._get_provider_limits)
        schema = limits.get("response_schema")
        options = {"response_schema": schema} if schema else {}
        count = self._auxiliary(role, "count_tokens",
            {"model": self.config["model"], "prompt_hash": digest([system, prompt, options]),
             **self._token_count_metadata()},
            lambda: self._count_input_tokens(prompt, system, **options))
        if count > limits["max_input_tokens"]:
            raise ValueError(f"{role} input uses {count} tokens; limit is {limits['max_input_tokens']}.")
        provider_input = self.provider_limits.get("input_token_limit")
        provider_output = self.provider_limits.get("output_token_limit")
        if type(provider_input) is not int or provider_input <= 0:
            raise ValueError("Provider context limit is unavailable. Set a verified model_context_tokens value.")
        if count + limits["max_output_tokens"] > provider_input:
            raise ValueError("Input plus output reserve exceeds the provider context limit.")
        if provider_output is not None and limits["max_output_tokens"] > provider_output:
            raise ValueError("Output reserve exceeds the provider output limit.")
        return count

    def call(self, role, prompt, system):
        identity = [self.provider, role, prompt, system, self.config.get("model"),
                    self._cache_settings()]
        limits = self.limits_for_role(role, prompt)
        identity.append(limits["max_output_tokens"])
        schema = limits.get("response_schema")
        if schema:
            identity.append(schema)
        key = digest(identity)
        with self.request_locks_guard:
            lock = self.request_locks.setdefault(key, threading.Lock())
        with lock:
            cached = self.store.get("api_response:" + key)
            if cached is not None:
                return cached["response"]
            result = self._call(role, prompt, system, limits)
            self.store.set("api_response:" + key, {"response": result})
            return result

    def _pacing_event(self, kind, **details):
        with self.cooldown_lock:
            pacing = {"effective_parallel_calls": self.gate.limit,
                      "max_parallel_calls": self.gate.maximum,
                      "not_before_utc": time.time() + max(0., self.not_before - time.monotonic())}
        self.store.set("api_pacing", pacing)
        write_json(self.run_dir / "api_events" / f"{time.time_ns()}-{kind}.json",
                   {"kind": kind, "utc": time.time(), **pacing, **details})

    def _wait_for_cooldown(self):
        """Observe extensions made by other requests before dispatch.

        Sleep in short chunks. Only a new shared deadline starts another wait;
        this also keeps tests with a mocked sleep from spinning indefinitely.
        """
        with self.cooldown_lock:
            deadline = self.not_before
        remaining = max(0., deadline - time.monotonic())
        while True:
            while remaining > 0:
                chunk = min(30., remaining)
                time.sleep(chunk)
                remaining -= chunk
            with self.cooldown_lock:
                extended = self.not_before
            if extended <= deadline:
                return
            remaining = max(0., extended - max(deadline, time.monotonic()))
            deadline = extended

    def _call(self, role, prompt, system, limits):
        config = self.config
        if "max_input_tokens" not in limits and len(prompt) + len(system) > limits["context_chars"]:
            raise ValueError("Prompt exceeds the context budget. Remove old observations first.")
        input_tokens = None
        for attempt in range(config.get("max_retries", 2) + 1):
            try:
                with self.gate:
                    self._wait_for_cooldown()
                    if "max_input_tokens" in limits and input_tokens is None:
                        input_tokens = self._check_token_limit(role, prompt, system, limits)
                    self._wait_for_cooldown()
                    result = self._attempt(role, prompt, system, limits, input_tokens)
                    old_limit = self.gate.limit
                    if self.gate.succeed() != old_limit:
                        self._pacing_event("recovered", role=role)
                    return result
            except Exception as error:
                if isinstance(error, AuthenticationUnavailable):
                    if not error.retryable or attempt >= config.get("max_retries", 2):
                        raise
                    self._pacing_event("authentication_retry", role=role, attempt=attempt + 1,
                                       dispatch_started=False)
                    time.sleep(min(4., 2. ** attempt))
                    continue
                code = getattr(error, "code", None)
                if code == 429:
                    self.gate.throttle()
                if code not in (429, 500, 502, 503, 504) or attempt >= config.get("max_retries", 2):
                    if code == 429:
                        self._pacing_event("retry_exhausted", role=role, attempt=attempt + 1, code=code)
                    raise
                base = config.get("retry_base_seconds", 30.)
                maximum = config.get("retry_max_seconds", 120.)
                delay = min(maximum, base * (2 ** attempt))
                delay = min(maximum, delay * random.uniform(1., 1.25))
                with self.cooldown_lock:
                    self.not_before = max(self.not_before, time.monotonic() + delay)
                self._pacing_event("retry_scheduled", role=role, attempt=attempt + 1,
                                   code=code, delay_seconds=delay)

    def _attempt(self, role, prompt, system, limits=None, input_tokens=None):
        limits = limits or self.limits_for_role(role)
        # Adaptive calls count inputs. Legacy roles can use a conservative byte bound.
        # Keep output headroom until the provider reports total usage.
        reserve = (input_tokens if input_tokens is not None else len((prompt + system).encode())) + limits["max_output_tokens"]
        call_id = self.store.reserve(role, reserve, self.config)
        path = self.run_dir / "calls" / f"{call_id:05d}.json"
        record = {"role": role, "provider": self.provider, "system": system, "prompt": prompt, "status": "pending",
                  "input_tokens": input_tokens, "reserved_tokens": reserve}
        if limits.get("response_schema"):
            record["response_schema"] = limits["response_schema"]
        write_json(path, record)
        tokens, text, usage = None, None, None
        try:
            text, usage = self._request(prompt, system, limits)
            tokens = usage.get("total_tokens") if isinstance(usage, dict) else usage
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                tokens = None
                raise ValueError("The provider returned an invalid total token count.")
            self.store.finish_call(call_id, tokens)
            write_json(path, {**record, "response": text, "tokens": tokens,
                              "usage": usage if isinstance(usage, dict) else {"total_tokens": tokens},
                              "model": self.config["model"], "status": "complete"})
            self._check_response(text, usage)
            return json.loads(text)
        except Exception as error:
            # Unknown outcomes retain their reservation. Quota rejections use zero tokens.
            before_dispatch = isinstance(error, AuthenticationUnavailable)
            rejected = before_dispatch or getattr(error, "generation_rejected", False) or getattr(error, "code", None) == 429
            self.store.finish_call(call_id, 0 if rejected else tokens, status="failed")
            write_json(path, {**record, "error": f"{type(error).__name__}: {error}",
                              "traceback": traceback.format_exc(), "response": text,
                              "usage": usage if isinstance(usage, dict) else {"total_tokens": tokens},
                              "tokens": 0 if rejected else tokens, "status": "failed",
                              "generation_rejected": rejected,
                              "dispatch_started": False if before_dispatch else None,
                              "failure_phase": "authentication" if before_dispatch else "generation_or_response"})
            raise


class Gemini(Agent):
    """Gemini on Vertex AI. Google credentials are loaded only for this provider."""
    provider = "vertex"

    def __init__(self, config, store, run_dir):
        super().__init__(config, store, run_dir)
        self.credentials_lock = threading.Lock()
        self.credentials = None

    def _credentials_for_client(self):
        """Refresh once under a lock before creating a provider client."""
        import datetime
        import google.auth
        from google.auth.exceptions import TransportError, RefreshError, DefaultCredentialsError
        from google.auth.transport.requests import Request
        class BoundedRequest(Request):
            def __call__(self, *args, **kwargs):
                kwargs["timeout"] = 30
                return super().__call__(*args, **kwargs)
        with self.credentials_lock:
            try:
                if self.credentials is None:
                    self.credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                expiry = self.credentials.expiry
                now = datetime.datetime.now(datetime.timezone.utc)
                if expiry is not None and expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=datetime.timezone.utc)
                if not self.credentials.valid or expiry is not None and expiry <= now + datetime.timedelta(minutes=5):
                    request = BoundedRequest()
                    try:
                        self.credentials.refresh(request)
                    finally:
                        request.session.close()
                return self.credentials
            except (TransportError, RefreshError, DefaultCredentialsError) as error:
                # Do not log credential/token contents or infer a generation was sent.
                raise AuthenticationUnavailable("Google authentication failed before generation dispatch: " + type(error).__name__,
                                                retryable=isinstance(error, TransportError)) from None

    def _client(self):
        from google import genai
        from google.genai import types
        config = self.config
        return genai.Client(vertexai=True, project=config["project"], location=config["location"],
            credentials=self._credentials_for_client(),
            http_options=types.HttpOptions(timeout=config["timeout_seconds"] * 1000,
                                           retry_options=types.HttpRetryOptions(attempts=1)))

    def _request(self, prompt, system, limits=None):
        from google.genai import types
        limits = limits or self.config
        client = self._client()
        try:
            response = client.models.generate_content(model=self.config["model"], contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system,
                    response_mime_type="application/json", temperature=1.0,
                    response_schema=limits.get("response_schema"),
                    max_output_tokens=limits["max_output_tokens"],
                    thinking_config=types.ThinkingConfig(thinking_level="LOW")))
            usage = response.usage_metadata
            return response.text or "", ({
                "total_tokens": getattr(usage, "total_token_count", None),
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "reasoning_tokens": getattr(usage, "thoughts_token_count", None),
                "cached_input_tokens": getattr(usage, "cached_content_token_count", None),
            } if usage else None)
        finally:
            client.close()

    def _count_input_tokens(self, prompt, system, response_schema=None):
        from google.genai import types
        client = self._client()
        try:
            generation = (types.GenerationConfig(response_schema=response_schema, response_mime_type="application/json")
                          if response_schema else None)
            response = client.models.count_tokens(model=self.config["model"], contents=prompt,
                config=types.CountTokensConfig(system_instruction=system, generation_config=generation))
            count = response.total_tokens
            if type(count) is not int or count < 0:
                raise ValueError("The provider did not return a valid input token count.")
            return count
        finally:
            client.close()

    def _get_provider_limits(self):
        client = self._client()
        try:
            model = client.models.get(model=self.config["model"])
            return {"input_token_limit": model.input_token_limit, "output_token_limit": model.output_token_limit}
        finally:
            client.close()
