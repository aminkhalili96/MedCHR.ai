from pathlib import Path

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.db import get_conn


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    schema_path = base / "sql" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")

    with get_conn() as conn:
        conn.execute(sql)
        conn.commit()

    print("DB schema applied")


if __name__ == "__main__":
    main()
