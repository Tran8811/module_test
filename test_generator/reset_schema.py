# reset_schema.py -- xoá 3 bảng test cũ (an toàn nếu chỉ là dữ liệu test)
# rồi tạo lại đúng theo pg_schema.sql

import psycopg2
from test_generator.pg_config import PG_CONN_PARAMS

conn = psycopg2.connect(**PG_CONN_PARAMS)
with conn:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS node_links CASCADE;")
        cur.execute("DROP TABLE IF EXISTS nodes CASCADE;")
        cur.execute("DROP TABLE IF EXISTS documents CASCADE;")

    with open("test_generator/pg_schema.sql", "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)

conn.close()
print("Đã xoá bảng cũ và tạo lại schema sạch!")