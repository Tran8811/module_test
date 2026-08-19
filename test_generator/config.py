CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100

SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".txt"
]
LLM_URL = "http://10.0.128.19:8010/v1/chat/completions"

MODEL_NAME = "llm-model"

# Repo HuggingFace của model thật đang chạy sau LLM_URL, dùng để load đúng
# tokenizer cho việc đếm token chính xác (thay vì ước lượng theo ký tự).
TOKENIZER_PATH = "google/gemma-4-26B-A4B-it"

TEMPERATURE = 0.2

MAX_TOKENS = 4000

CANDIDATE_PREFILTER_LIMIT = 100

MODEL_CONTEXT_WINDOW = 65536
CONTEXT_WINDOW_SAFETY_MARGIN = 1024

CANDIDATE_CHUNK_SNIPPET_TOKENS = 150