import psycopg2
from test_generator.pg_config import PG_CONN_PARAMS

with open("test_generator/pg_schema.sql", "r", encoding="utf-8") as f:
    sql = f.read()

conn = psycopg2.connect(**PG_CONN_PARAMS)
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
conn.close()
print("Tạo schema xong!")