import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.structured_normalize import normalize_structured


def main() -> None:
    result = normalize_structured(write_csv=True)
    print(json.dumps({
        "dataset_root": result["dataset_root"],
        "counts": result["counts"],
        "errors": result["errors"],
    }, indent=2))


if __name__ == "__main__":
    main()
