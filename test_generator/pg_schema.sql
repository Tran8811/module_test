-- pg_schema.sql
--  Chạy 1 lần trên container pgvector/pgvector:0.8.6-pg17-trixie
-- trước khi dùng pg_writer.py / pg_reader.py.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          VARCHAR(36) PRIMARY KEY,
    bot_id      VARCHAR(36),
    file_name   TEXT,
    active      BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS nodes (
    id           VARCHAR(36) PRIMARY KEY,
    document_id  VARCHAR(36) REFERENCES documents(id),
    content      TEXT NOT NULL,
    embedding    VECTOR(3072),   -- NULL với node cha, có giá trị với node lá
    tsv          TSVECTOR
);

CREATE INDEX IF NOT EXISTS ix_nodes_document_id ON nodes (document_id);
CREATE INDEX IF NOT EXISTS ix_nodes_tsv ON nodes USING gin (tsv);
-- Khi dữ liệu đủ lớn có thể bật thêm index ivfflat/hnsw cho embedding:
-- CREATE INDEX ix_nodes_embedding ON nodes USING ivfflat (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS node_links (
    parent_id    VARCHAR(36) REFERENCES nodes(id),
    child_id     VARCHAR(36) REFERENCES nodes(id),
    order_index  INTEGER NOT NULL,
    PRIMARY KEY (parent_id, child_id, order_index)
);

CREATE INDEX IF NOT EXISTS ix_node_links_parent_id ON node_links (parent_id);
CREATE INDEX IF NOT EXISTS ix_node_links_child_id ON node_links (child_id);
CREATE INDEX IF NOT EXISTS ix_node_links_parent_order ON node_links (parent_id, order_index);
