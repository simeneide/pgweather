# %%
# Write table
from __future__ import annotations

from typing import Optional

import polars as pl

from src.web.config import settings


def _get_database_uri() -> str:
    return settings.db_url


class Database:
    """Simple wrapper around polars to read and write to Postgres."""

    def __init__(self, uri: Optional[str] = None):
        self.uri = uri or _get_database_uri()

    def read(self, query: str) -> pl.DataFrame:
        return pl.read_database_uri(query=query, uri=self.uri, engine="adbc")

    def write(
        self,
        df: pl.DataFrame,
        table_name: str,
        if_table_exists: str = "append",
    ) -> bool:
        df.write_database(
            table_name=table_name,
            connection=self.uri,
            engine="adbc",
            if_table_exists=if_table_exists,  # type: ignore[arg-type]
        )
        return True

    def replace_data(
        self,
        df: pl.DataFrame,
        table_name: str,
        model_source: Optional[str] = None,
    ) -> bool:
        """Replace data in *table_name*, optionally scoped to *model_source*.

        When *model_source* is given, only rows matching that source are
        deleted (allowing data from other models to coexist).  When ``None``,
        the table is truncated entirely (legacy behaviour).

        Falls back to ``if_table_exists="replace"`` if the table doesn't exist.
        """
        table_exists = True
        try:
            if model_source is not None:
                import adbc_driver_postgresql.dbapi as pg_dbapi

                with pg_dbapi.connect(self.uri) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"DELETE FROM {table_name} WHERE model_source = '{model_source}'"
                        )
                    conn.commit()
            else:
                self.execute_query(f"TRUNCATE TABLE {table_name}")
        except Exception:
            # Table may not exist yet — let write_database create it
            table_exists = False

        df.write_database(
            table_name=table_name,
            connection=self.uri,
            engine="adbc",
            if_table_exists="append" if table_exists else "replace",
        )
        return True

    def replace_forecast(
        self,
        df: pl.DataFrame,
        table_name: str,
        forecast_timestamp: str,
        model_source: str = "meps",
    ) -> bool:
        """Replace rows for a specific (forecast_timestamp, model_source).

        Deletes existing rows matching *forecast_timestamp* **and**
        *model_source* then appends *df*.  Falls back to creating the table
        if it doesn't exist yet.
        """
        import adbc_driver_postgresql.dbapi as pg_dbapi

        table_exists = True
        try:
            with pg_dbapi.connect(self.uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {table_name} "
                        f"WHERE forecast_timestamp = '{forecast_timestamp}' "
                        f"AND model_source = '{model_source}'"
                    )
                conn.commit()
        except Exception:
            table_exists = False

        df.write_database(
            table_name=table_name,
            connection=self.uri,
            engine="adbc",
            if_table_exists="append" if table_exists else "replace",
        )
        return True

    def execute_query(self, query: str, timeout_ms: int = 300_000) -> list | None:
        """Execute a raw SQL query with an explicit statement timeout.

        *timeout_ms* defaults to 5 minutes (Supabase free-tier default is
        often much shorter and causes DDL operations to fail).
        """
        import psycopg2

        conn = psycopg2.connect(self.uri)
        cur = conn.cursor()
        cur.execute(f"SET statement_timeout = '{timeout_ms}'")
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
