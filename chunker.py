from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = splitter.split_documents(documents)

    chunks = []

    for i, doc in enumerate(docs):
        chunks.append(
            {
                "chunk_id": i,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return chunks