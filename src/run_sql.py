"""
Runs sql/analysis.sql against the CSVs with DuckDB and prints each result set.

No database server needed — DuckDB reads the CSV files directly, so the SQL in
this repo is real, runnable SQL rather than a text file nobody ever executed.

Run:  python src/run_sql.py
"""

import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQL = ROOT / "sql" / "analysis.sql"

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 30)


def split_statements(text: str):
    """Split on semicolons that end a statement, keeping the leading comments."""
    chunks, buf = [], []
    for line in text.splitlines():
        buf.append(line)
        if line.strip().endswith(";"):
            chunks.append("\n".join(buf))
            buf = []
    if buf:
        chunks.append("\n".join(buf))
    return [c for c in chunks if c.strip() and not c.strip().startswith("--\n")]


def title_of(chunk: str) -> str:
    for line in chunk.splitlines():
        s = line.strip().lstrip("- ").strip()
        if s and re.match(r"^Q\d", s):
            return s
    return ""


def main():
    con = duckdb.connect()
    con.execute(f"SET FILE_SEARCH_PATH='{ROOT}'")
    for chunk in split_statements(SQL.read_text()):
        sql = "\n".join(l for l in chunk.splitlines() if not l.strip().startswith("--")).strip()
        if not sql:
            continue
        name = title_of(chunk)
        try:
            result = con.execute(sql)
            if sql.upper().startswith("CREATE"):
                continue
            df = result.fetch_df()
            print("\n" + "=" * 100)
            print(name or sql.splitlines()[0][:80])
            print("=" * 100)
            print(df.to_string(index=False))
        except Exception as exc:                      # noqa: BLE001
            print(f"\n!! failed: {name}\n   {exc}")
    print()


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    main()
