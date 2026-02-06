import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.migrate import main as migrate_main


def main() -> None:
    migrate_main()


if __name__ == "__main__":
    main()
