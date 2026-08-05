QUESTION_PROMPT = """
You are creating a benchmark dataset for evaluating Retrieval systems.

Given one document chunk below.

Generate exactly 3 questions that can ONLY be answered from this chunk.

Requirements:

- Questions must be factual.
- Questions should not depend on external knowledge.
- Avoid yes/no questions.
- Avoid ambiguous wording.
- Different questions should test different information.

Return ONLY valid JSON.

Format:

{{
    "questions": [
        "...",
        "...",
        "..."
    ]
}}

Chunk:

{chunk}
"""
LABEL_PROMPT = """
You are evaluating candidate chunks for a Retrieval benchmark.

Question:
{question}

Candidate chunks:
{chunks}

Assign one relevance score for every candidate.

Relevance:
3 = Contains the complete answer.
2 = Contains important supporting information.
1 = Slightly relevant.
0 = Irrelevant.

Rules:
- Return ONLY one JSON object.
- Do NOT explain.
- Do NOT use markdown.
- Do NOT wrap with ```json.
- chunk_id MUST be an integer.
- Return every candidate exactly once.

Output format:

{{
  "results": [
    {{
      "chunk_id": 2,
      "relevance": 3
    }}
  ]
}}
"""

CANDIDATE_PROMPT = """
You are building a Retrieval benchmark.

Question:
{question}

Below are document chunks.

Select the {top_k} most relevant chunks.

Rules:
- Return ONLY one JSON object.
- Do NOT explain.
- Do NOT use markdown.
- Do NOT wrap with ```json.
- Do NOT output any text before or after the JSON.
- Return ONLY integer chunk IDs.
- If a chunk partially answers the question, include it.
- Prefer chunks containing direct evidence.

Chunks:
{chunks}

Output format:

{{
  "candidate_chunks": [1, 5, 9]
}}
"""
ANSWER_PROMPT = """
You are creating a benchmark dataset for Question Answering.

Question:

{question}

Relevant Chunks:

{chunks}

Requirements:

- Answer ONLY using the provided chunks.
- Do not use external knowledge.
- If information is insufficient, answer "Not enough information."
- Keep the answer concise and factual.

Return ONLY valid JSON.

{{
    "answer":"..."
}}
"""