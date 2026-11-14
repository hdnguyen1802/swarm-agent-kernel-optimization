"""DeepSeek Chat Completions with the shared search budget and no Google SDK."""

from http.client import HTTPSConnection
import json
import os

from .agent import Agent, AuthenticationUnavailable


class DeepSeekRequestError(RuntimeError):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code
        # A documented rejection is not an uncertain generation outcome.
        self.generation_rejected = code in {400, 401, 402, 403, 404, 422, 429}


def _json_schema(value):
    if isinstance(value, dict):
        return {k: v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)
                for k, v in value.items()}
    return [_json_schema(v) for v in value] if isinstance(value, list) else value


class DeepSeek(Agent):
    provider = "deepseek"

    def limits_for_role(self, role, prompt=None):
        limits = super().limits_for_role(role, prompt)
        limits.setdefault("max_input_tokens", self.config["model_context_tokens"])
        return limits

    def _messages(self, prompt, system, response_schema=None):
        instructions = system + "\nReturn one JSON object, without Markdown fences."
        if response_schema:
            instructions += "\nUse this response structure:\n" + json.dumps(_json_schema(response_schema), ensure_ascii=False)
        return [{"role": "system", "content": instructions}, {"role": "user", "content": prompt}]

    def _token_count_metadata(self):
        return {"method": "utf8_byte_bound", "remote_request": False,
                "note": "Conservative input reservation, including JSON instructions and framing headroom. "
                        "The response usage supplies actual token counts."}

    def _count_input_tokens(self, prompt, system, response_schema=None):
        # DeepSeek has no public count_tokens endpoint. Avoid unsafe chars/4 estimates.
        messages = self._messages(prompt, system, response_schema)
        return len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 256

    def _get_provider_limits(self):
        raise ValueError("Set verified DeepSeek model_context_tokens and model_output_tokens in the config.")

    def _cache_settings(self):
        return {"thinking": self.config.get("thinking", "enabled"),
                "reasoning_effort": self.config.get("reasoning_effort", "low")}

    def _request(self, prompt, system, limits=None):
        limits = limits or self.config
        variable = self.config.get("api_key_env", "DEEPSEEK_API_KEY")
        key = os.environ.get(variable, "").strip()
        if not key or not key.isascii() or any(c.isspace() for c in key):
            raise AuthenticationUnavailable("Set a valid DeepSeek API key in " + variable + ".", retryable=False)
        thinking = self.config.get("thinking", "enabled")
        body = {"model": self.config["model"],
                "messages": self._messages(prompt, system, limits.get("response_schema")),
                "response_format": {"type": "json_object"}, "max_tokens": limits["max_output_tokens"],
                "stream": False, "thinking": {"type": thinking}}
        if thinking == "enabled":
            body["reasoning_effort"] = self.config.get("reasoning_effort", "low")
        connection = None
        try:
            connection = HTTPSConnection("api.deepseek.com", timeout=self.config["timeout_seconds"])
            connection.request("POST", "/chat/completions", body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                               headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                # Error bodies can echo credentials or arbitrary request fields. Do not log them.
                raise DeepSeekRequestError("DeepSeek returned HTTP " + str(response.status) + ".", response.status)
            payload = json.loads(response.read())
            if not isinstance(payload, dict):
                raise ValueError("Expected a response object.")
            raw = payload.get("usage")
            usage = {"total_tokens": None}
            if isinstance(raw, dict):
                details = raw.get("completion_tokens_details") or {}
                usage = {"total_tokens": raw.get("total_tokens"), "input_tokens": raw.get("prompt_tokens"),
                         "output_tokens": raw.get("completion_tokens"),
                         "reasoning_tokens": details.get("reasoning_tokens") if isinstance(details, dict) else None,
                         "cached_input_tokens": raw.get("prompt_cache_hit_tokens")}
            choices = payload.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = choice.get("message") or {}
            text = message.get("content") if isinstance(message, dict) else None
            usage["finish_reason"] = choice.get("finish_reason")
            return text if isinstance(text, str) else "", usage
        except DeepSeekRequestError:
            raise
        except Exception as error:
            raise DeepSeekRequestError("DeepSeek request failed: " + type(error).__name__ + ".") from None
        finally:
            if connection is not None:
                connection.close()

    def _check_response(self, text, usage):
        if isinstance(usage, dict) and usage.get("finish_reason") not in {None, "stop"}:
            raise ValueError("DeepSeek did not finish the JSON response: " + str(usage["finish_reason"]) + ".")
