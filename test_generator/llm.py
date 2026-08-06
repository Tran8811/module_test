import time
import requests
from typing import Any, Dict

from .config import *

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy"
}

# Rough characters-per-token ratio. Word-splitting badly UNDER-estimates
# token count for Vietnamese (diacritics/subword tokenization commonly
# produce 1.5-3 tokens per word), which was the real cause of prompts
# silently passing the local check and still getting rejected by the
# server -> falling back to a near-empty prompt -> missing candidates.
# A char-based estimate is safer and doesn't need any network call
# (unlike tiktoken, which needs to download encoding files on first use
# and will fail on machines without internet access, like an internal
# LLM host such as 10.0.128.19).
CHARS_PER_TOKEN = 2.2  # conservative for Vietnamese; tune with real usage data below


def _estimate_tokens(text: str) -> int:
    """Character-based token estimate (no network dependency).

    This intentionally over-estimates slightly rather than
    under-estimating: it's much cheaper to trim one extra chunk than to
    silently drop the chunk that actually contains the answer.
    """
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _estimate_payload_tokens(prompt: str, extra: Dict[str, Any] | None = None) -> int:
    t = _estimate_tokens(prompt)
    overhead = 50  # JSON/message wrapper overhead
    if extra:
        for v in extra.values():
            if isinstance(v, str):
                t += _estimate_tokens(v)
    return t + overhead


def chat(prompt: str, max_retries: int = 5, backoff_seconds: float = 5.0):
    input_tokens = _estimate_payload_tokens(prompt)
    total_estimated = input_tokens + MAX_TOKENS

    allowed = MODEL_CONTEXT_WINDOW - CONTEXT_WINDOW_SAFETY_MARGIN
    if total_estimated > allowed:
        raise requests.HTTPError(
            f"ContextWindowExceededError: estimated input_tokens={input_tokens}, "
            f"max_output_tokens={MAX_TOKENS}, total_estimated={total_estimated}. "
            f"Model context window={MODEL_CONTEXT_WINDOW} (allowed ~{allowed}). "
            "Reduce the prompt size or lower MAX_TOKENS, or use candidate prefiltering."
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                LLM_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            body = response.text if response is not None else ""
            status = response.status_code if response is not None else None
            last_exc = requests.HTTPError(
                f"{exc}\nLLM request payload length={len(str(payload))}, response body={body}"
            )
            # 503 = pod not up yet. 500 here means the inference engine
            # itself crashed mid-request (seen in practice with very large
            # candidate-selection prompts) -- the pod may come back after
            # a restart, so both are worth retrying with a longer backoff
            # than a simple context/payload problem would need.
            if status in (500, 503) and attempt < max_retries:
                wait = backoff_seconds * (2 ** (attempt - 1))  # exponential: 5s, 10s, 20s...
                print(f"[llm-retry] status={status} attempt={attempt}/{max_retries}, waiting {wait:.0f}s before retry")
                time.sleep(wait)
                continue
            raise last_exc from exc
        except requests.exceptions.RequestException as exc:
            # network-level errors (timeout, connection refused, etc.) -> also transient
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
                continue
            raise

        data = response.json()

        # Calibrate our estimate against the real token usage the server
        # reports, when available, so CHARS_PER_TOKEN can be tuned later
        # instead of guessed blindly.
        usage = data.get("usage")
        if usage and usage.get("prompt_tokens"):
            real_tokens = usage["prompt_tokens"]
            ratio = len(prompt) / real_tokens if real_tokens else None
            if ratio:
                print(
                    f"[token-calibration] estimated={input_tokens} real={real_tokens} "
                    f"implied_chars_per_token={ratio:.2f} (current constant={CHARS_PER_TOKEN})"
                )

        return data["choices"][0]["message"]["content"]

    # Should not reach here, but just in case.
    raise last_exc