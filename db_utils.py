# %%
# Write table
import polars as pl
import streamlit as st
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
import os


def _get_database_uri() -> str:
    database_url = os.getenv("SUPABASE_DB_URL")
    if database_url:
        return database_url

    # Backward-compatible fallback for legacy Aiven credentials.
    aiven_user = os.getenv("AIVEN_USER")
    aiven_password = os.getenv("AIVEN_PASSWORD")
    if aiven_user and aiven_password:
        return (
            "postgres://"
            f"{aiven_user}:{aiven_password}"
            "@pg-weather-pg-weather.b.aivencloud.com:20910/defaultdb"
            "?sslmode=require"
        )

    raise RuntimeError("Database credentials missing. Set SUPABASE_DB_URL.")


class Database:
    """Simple wrapper around polars to read and write to Postgres."""

    def __init__(self, uri: Optional[str] = None):
        self.uri = uri or _get_database_uri()

    def read(self, query):
        """
        query = "SELECT * FROM weather_stations"
        query = "SELECT * FROM weather_measurements"
        """

        df = pl.read_database_uri(query=query, uri=self.uri, engine="adbc")
        return df

    def write(self, df: pl.DataFrame, table_name, if_table_exists="append"):
        df.write_database(
            table_name=table_name,
            connection=self.uri,
            engine="adbc",
            if_table_exists=if_table_exists,
        )
        return True

    def execute_query(self, query):
        """
        query = "select * from records"
        """
        import psycopg2

        conn = psycopg2.connect(self.uri)
        cur = conn.cursor()
        cur.execute(query)
        result = None
        try:
            result = cur.fetchall()
        except psycopg2.ProgrammingError:
            # If no results are to be fetched, e.g., for INSERT, UPDATE, etc.
            conn.commit()

        cur.close()
        conn.close()
        return result


if __name__ == "__main__":
    db = Database()
    db.read("SELECT * FROM weather_measurements")
    db.write(pl.DataFrame({"bs": [1, 2, 4], "kake": ["hei", "på", "deg"]}), "records")

    # %% add indexes
    if False:
        create_index_query = (
            "CREATE INDEX idx_time_name ON weather_measurements (time, name);"
        )
        res = db.execute_query(create_index_query)
        create_index_query = "CREATE INDEX idx_time ON weather_measurements (time);"
        res = db.execute_query(create_index_query)
        res
    index_query = """
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE schemaname = 'public';
            """
    res = db.execute_query(index_query)
    print(res)
    # %% test query
    res = db.execute_query(
        "EXPLAIN SELECT * FROM weather_measurements where time > '2025-03-03 08:47:28'"
    )
# %%
