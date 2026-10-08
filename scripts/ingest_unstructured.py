import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ingestion_service import run_ingestion


def main() -> None:
    result = run_ingestion()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
