# pg_config.py
"""
Thông tin kết nối tới container pgvector/pgvector:0.8.6-pg17-trixie.

    POSTGRES_HOSTNAME=localhost
    POSTGRES_PORT=5432
    POSTGRES_USERNAME=postgres
    POSTGRES_PASSWORD=...
    POSTGRES_NAME=personal_assistance
"""
import os

PG_CONN_PARAMS = {
    "host": os.environ.get("POSTGRES_HOSTNAME", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "user": os.environ.get("POSTGRES_USERNAME", "myuser"),
    "password": os.environ.get("POSTGRES_PASSWORD", "mypassword"),
    "dbname": os.environ.get("POSTGRES_NAME", "trolyao"),
    "client_encoding": "utf8",
}
