# Legacy character-based sizes (kept for backward compatibility)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Token-based chunking (preferred). These are approximate token counts
# used to avoid exceeding LLM context limits. Token counting uses a
# simple word-based approximation; replace with a tokenizer if desired.
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100

SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".txt"
]
LLM_URL = "http://10.0.128.19:8010/v1/chat/completions"

MODEL_NAME = "llm-model"

TEMPERATURE = 0.2

MAX_TOKENS = 4000

CANDIDATE_PREFILTER_LIMIT = 100

MODEL_CONTEXT_WINDOW = 65536
# Safety margin reserved for system prompts and unexpected overhead.
CONTEXT_WINDOW_SAFETY_MARGIN = 1024

CANDIDATE_CHUNK_SNIPPET_TOKENS = 150