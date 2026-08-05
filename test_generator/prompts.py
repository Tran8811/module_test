QUESTION_PROMPT = """
You are creating a benchmark dataset for evaluating Retrieval and QA systems.

Given one document chunk below, generate exactly 3 questions with types and difficulty labels.

Requirements:
- Questions must be factual.
- Questions should not depend on external knowledge.
- Avoid yes/no questions.
- Avoid ambiguous wording.
- Different questions should test different information.
- If the chunk contains a table or structured data, include one table-related question.

Return ONLY valid JSON.

Format:
{{
  "questions": [
    {{
      "question": "...",
      "type": "one-hop|multi-hop|table",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Type definitions:
- one-hop: can be answered directly using one fact or one chunk.
- multi-hop: requires reasoning across multiple facts or combining information.
- table: based on tabular or structured data in the chunk.

Difficulty definitions:
- easy: direct factual retrieval.
- medium: requires a moderate inference or combination of facts.
- hard: requires deeper reasoning or multi-step synthesis.

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
- Return only candidates with relevance > 0.
- Do NOT include chunks with relevance = 0 in the results.
- Assume any candidate not listed has relevance = 0.

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