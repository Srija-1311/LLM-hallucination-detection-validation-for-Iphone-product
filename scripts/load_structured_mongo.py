import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.structured_loader import load_structured_mongo


def main() -> None:
    result = load_structured_mongo()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
