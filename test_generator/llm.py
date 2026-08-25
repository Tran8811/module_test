import json
import time
import requests
from typing import Any, Dict

from .config import *

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy"
}

# Fallback nếu không load được tokenizer thật.
CHARS_PER_TOKEN = 2.2

BASE_TIMEOUT_SECONDS = 120
SECONDS_PER_OUTPUT_TOKEN = 0.05

# Gemma 4 hỗ trợ "thinking mode" (sinh reasoning_content trước khi sinh
# content thật). Với các task chỉ cần output JSON đơn giản (phân loại
# heading, chọn candidate...), thinking không cần thiết và có nguy cơ
# ngốn hết max_tokens vào phần suy nghĩ, khiến content thật bị cắt cụt
# hoặc rỗng. Thử tắt qua chat_template_kwargs (convention phổ biến với
# vLLM/SGLang cho các model hỗ trợ enable_thinking). Nếu server không hỗ
# trợ tham số này, nó thường được bỏ qua thay vì lỗi -- cần xác nhận qua
# log [llm-reasoning] xem có còn xuất hiện reasoning_content hay không.
DISABLE_THINKING = True

_tokenizer = None
_tokenizer_load_failed = False


def _get_tokenizer():
    global _tokenizer, _tokenizer_load_failed

    if _tokenizer is not None or _tokenizer_load_failed:
        return _tokenizer

    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as exc:
        _tokenizer_load_failed = True
        print(
            f"[token-count] Không load được tokenizer từ TOKENIZER_PATH="
            f"{TOKENIZER_PATH!r} ({exc}). Fallback sang ước lượng theo "
            f"CHARS_PER_TOKEN={CHARS_PER_TOKEN}."
        )
        return None

    return _tokenizer


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0

    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return int(len(text) / CHARS_PER_TOKEN) + 1


def _estimate_payload_tokens(
    prompt: str,
    extra: Dict[str, Any] | None = None
) -> int:
    t = _estimate_tokens(prompt)
    overhead = 50

    if extra:
        for v in extra.values():
            if isinstance(v, str):
                t += _estimate_tokens(v)

    return t + overhead


def _stream_chat_completion(payload: dict, request_timeout: float):
    """Gọi endpoint với stream=True để tránh bị gateway/proxy (LiteLLM) tự
    ngắt kết nối với lỗi 504/408 khi model cần lâu để sinh xong output dài.

    Trả về (content, reasoning, finish_reason, usage). Tách riêng content
    (câu trả lời thật) và reasoning (nội dung "suy nghĩ" nếu model có
    thinking mode) để không bị nhầm lẫn khi debug output_len quá ngắn.
    """
    stream_payload = {**payload, "stream": True}

    response = requests.post(
        LLM_URL,
        headers=headers,
        json=stream_payload,
        timeout=request_timeout,
        stream=True,
    )
    response.raise_for_status()

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    usage = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue

        data_str = raw_line[len("data:"):].strip()
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {}) or {}

            piece = delta.get("content")
            if piece:
                content_parts.append(piece)

            # Một số server (vLLM + reasoning-parser gemma4, hoặc tương
            # đương) trả phần "suy nghĩ" ở field riêng, không lẫn vào
            # content. Bắt lại để không mất token vào chỗ không rõ ràng.
            reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        if chunk.get("usage"):
            usage = chunk["usage"]

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)

    if reasoning and not content:
        print(
            f"[llm-reasoning] Model chỉ sinh reasoning ({len(reasoning)} ký tự), "
            "chưa kịp sinh content thật trước khi hết max_tokens. "
            "Nếu lặp lại thường xuyên, cân nhắc tắt thinking mode hoặc tăng max_tokens."
        )
    elif reasoning:
        print(f"[llm-reasoning] Model sinh {len(reasoning)} ký tự reasoning trước khi trả content.")

    return content, reasoning, finish_reason, usage


def chat(
    prompt: str,
    max_retries: int = 5,
    backoff_seconds: float = 5.0,
    max_tokens: int | None = None,
    response_format: Dict[str, Any] | None = None,
):
    output_tokens = max_tokens if max_tokens is not None else MAX_TOKENS

    input_tokens = _estimate_payload_tokens(prompt)
    total_estimated = input_tokens + output_tokens

    allowed = MODEL_CONTEXT_WINDOW - CONTEXT_WINDOW_SAFETY_MARGIN

    if total_estimated > allowed:
        raise requests.HTTPError(
            f"ContextWindowExceededError: estimated input_tokens={input_tokens}, "
            f"max_output_tokens={output_tokens}, total_estimated={total_estimated}. "
            f"Model context window={MODEL_CONTEXT_WINDOW} (allowed ~{allowed}). "
            "Reduce the prompt size or lower max_tokens, or use candidate prefiltering."
        )

    request_timeout = max(BASE_TIMEOUT_SECONDS, output_tokens * SECONDS_PER_OUTPUT_TOKEN)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": output_tokens
    }

    if DISABLE_THINKING:
        # Convention phổ biến cho model hỗ trợ configurable thinking mode
        # khi serve qua vLLM/SGLang (áp dụng chat template với cờ tắt
        # thinking). Nếu server/model không hỗ trợ, field này thường bị
        # bỏ qua chứ không lỗi -- theo dõi log [llm-reasoning] để biết
        # có hiệu quả hay không.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    if response_format is not None:
        # Structured output kiểu OpenAI (response_format={"type": "json_schema",
        # "json_schema": {...}}), được vLLM (>=0.6, backend xgrammar/outlines)
        # và SGLang hỗ trợ native qua endpoint OpenAI-compatible. Model bị ép
        # sinh đúng theo schema ở tầng decoding, không chỉ "được dặn" trong
        # prompt -- nhờ vậy khỏi cần nhồi hướng dẫn format vào prompt, và kết
        # quả trả về gần như luôn là JSON hợp lệ (vẫn có thể bị cắt cụt nếu
        # chạm max_tokens, nên vẫn giữ các cơ chế vá JSON ở toc_detector.py).
        payload["response_format"] = response_format

    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            content, reasoning, finish_reason, usage = _stream_chat_completion(payload, request_timeout)

        except requests.exceptions.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            status = exc.response.status_code if exc.response is not None else None

            last_exc = requests.HTTPError(
                f"{exc}\n"
                f"LLM request payload length={len(str(payload))}, "
                f"response body={body}"
            )

            if status in (500, 503, 408, 504) and attempt < max_retries:
                wait = backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"[llm-retry] status={status} "
                    f"attempt={attempt}/{max_retries}, "
                    f"timeout_used={request_timeout:.0f}s, "
                    f"waiting {wait:.0f}s before retry"
                )
                time.sleep(wait)
                continue

            raise last_exc from exc

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"[llm-retry] client timeout sau {request_timeout:.0f}s "
                    f"attempt={attempt}/{max_retries}, waiting {wait:.0f}s before retry"
                )
                time.sleep(wait)
                continue
            raise

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
                continue
            raise

        if not content:
            last_exc = requests.HTTPError(
                f"Empty content from LLM (reasoning_len={len(reasoning)}, "
                f"finish_reason={finish_reason})"
            )
            if attempt < max_retries:
                wait = backoff_seconds * (2 ** (attempt - 1))
                print(f"[llm-retry] empty content, attempt={attempt}/{max_retries}, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            raise last_exc

        if usage and usage.get("prompt_tokens"):
            real_tokens = usage["prompt_tokens"]
            diff_pct = (
                abs(input_tokens - real_tokens) / real_tokens * 100
                if real_tokens else 0
            )
            print(
                f"[token-calibration] estimated={input_tokens} "
                f"real={real_tokens} "
                f"diff={diff_pct:.1f}%"
            )

        if finish_reason == "length":
            print(
                f"[llm-truncated] Response bị cắt do chạm max_tokens={output_tokens}. "
                f"input_tokens~={input_tokens}, output_len={len(content)} ký tự, "
                f"reasoning_len={len(reasoning)} ký tự."
            )

        return content

    raise last_exc