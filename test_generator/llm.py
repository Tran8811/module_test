import time
import requests
from typing import Any, Dict

from .config import *

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy"
}

CHARS_PER_TOKEN = 2.2


def _estimate_tokens(text: str) -> int:
    """Ước lượng số token dựa trên số lượng ký tự, không phụ thuộc vào network.

    Cách này cố tình ước lượng cao hơn một chút thay vì ước lượng thấp:
    việc cắt bỏ thêm một chunk sẽ ít tốn kém hơn so với việc vô tình bỏ qua
    chunk thực sự chứa câu trả lời.
    """
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _estimate_payload_tokens(
    prompt: str,
    extra: Dict[str, Any] | None = None
) -> int:
    t = _estimate_tokens(prompt)
    overhead = 50  # Phần overhead ước tính của wrapper JSON/message

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
                f"{exc}\n"
                f"LLM request payload length={len(str(payload))}, "
                f"response body={body}"
            )

            # 503 = pod chưa sẵn sàng.
            # 500 = inference engine đã bị crash giữa request
            # (thực tế có thể xảy ra khi prompt dùng để chọn candidate quá lớn).
            # Pod có thể hoạt động trở lại sau khi restart, vì vậy cả hai lỗi này
            # đều đáng để retry với thời gian chờ dài hơn so với lỗi thông thường
            # liên quan đến context/payload.
            if status in (500, 503) and attempt < max_retries:
                # Thời gian chờ tăng theo cấp số nhân: 5s, 10s, 20s, ...
                wait = backoff_seconds * (2 ** (attempt - 1))

                print(
                    f"[llm-retry] status={status} "
                    f"attempt={attempt}/{max_retries}, "
                    f"waiting {wait:.0f}s before retry"
                )

                time.sleep(wait)
                continue

            raise last_exc from exc

        except requests.exceptions.RequestException as exc:
            last_exc = exc

            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
                continue

            raise

        data = response.json()

        # Hiệu chỉnh lại phép ước lượng dựa trên số token thực tế
        # mà server trả về, nếu có, để sau này có thể điều chỉnh
        # CHARS_PER_TOKEN thay vì phải sử dụng một giá trị ước lượng cố định.
        usage = data.get("usage")

        if usage and usage.get("prompt_tokens"):
            real_tokens = usage["prompt_tokens"]
            ratio = len(prompt) / real_tokens if real_tokens else None

            if ratio:
                print(
                    f"[token-calibration] estimated={input_tokens} "
                    f"real={real_tokens} "
                    f"implied_chars_per_token={ratio:.2f} "
                    f"(current constant={CHARS_PER_TOKEN})"
                )

        return data["choices"][0]["message"]["content"]
    raise last_exc