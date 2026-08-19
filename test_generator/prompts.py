QUESTION_PROMPT = """
You are creating a benchmark dataset for evaluating Retrieval and QA systems.

Given one document chunk below (with its chunk id), generate exactly 3 questions with types and difficulty labels.

Requirements:
- Questions must be factual.
- Questions should not depend on external knowledge.
- Avoid yes/no questions.
- Avoid ambiguous wording.
- Different questions should test different information.
- If the chunk contains a table or structured data, include one table-related question.
- If related chunks are available, create at least one question that requires combining information from multiple chunks.

Self-contained requirement (very important):
- Each question MUST make full sense on its own, with NO reference to "the chunk", "this document",
  "the text above", "the passage", "as mentioned", "the given content", or any similar phrase that
  assumes the reader has already seen the chunk.
- Do NOT use vague pronouns ("it", "he", "this", "that") without a clear antecedent stated in the
  question itself. Spell out the actual subject (person name, organization, date, term...) explicitly.
- A question must be understandable and answerable by someone who has NEVER seen this chunk, and who
  will later search a large document collection to find the answer — it should read like a natural
  standalone question, not a reading-comprehension question about "this text".
- Bad example: "What is mentioned in the table above about the score?"
- Good example: "What is the average score of students in class 10A2 according to the semester report?"

For every question, you MUST also return which chunk id(s) contain the evidence needed to answer it:
- For "one-hop" and "table" questions: source_chunk_ids should contain only the id of the given chunk below.
- For "multi-hop" questions: source_chunk_ids should include the id of the given chunk below AND the id(s)
  of any related chunk(s) (from "Related chunks") that are actually needed to answer the question.

Return ONLY valid JSON.

Format:
{{
  "questions": [
    {{
      "question": "...",
      "type": "one-hop|multi-hop|table",
      "difficulty": "easy|medium|hard",
      "source_chunk_ids": ["..."]
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

Chunk id: {chunk_id}

Chunk:

{chunk}

Related chunks:

{related_chunks}
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
- If the question requires comparing or combining information, return all chunks needed to answer it fully.

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
- If the answer requires multiple chunks, combine them clearly.
- If information is insufficient, answer "Not enough information."
- Keep the answer concise and factual.

Return ONLY valid JSON.

{{
    "answer":"..."
}}
"""