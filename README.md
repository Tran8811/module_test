# test_generator

Generate retrieval and QA benchmark datasets from document chunks using an LLM.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- Optional: `pip install uv`

## Run

- `python main.py`
- or `uv run`

## Notes

- Input PDF: `data/Diem-CK_HP.pdf`
- Output: `output/retrieval_test.json`, `output/qa_test.json`
