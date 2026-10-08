import zipfile
from pathlib import Path

from app.config import DATASET_EXTRACT_DIR, DATASET_ROOT, DATASET_ZIP


def resolve_dataset_root() -> Path:
    """Return the extracted synthetic-data folder, unpacking the zip if needed."""
    if DATASET_ROOT.exists() and (DATASET_ROOT / "metadata").exists():
        return DATASET_ROOT

    nested = DATASET_ROOT / "foxconn synthetic data"
    if nested.exists() and (nested / "metadata").exists():
        return nested

    if not DATASET_ZIP.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at {DATASET_ROOT} and zip not found at {DATASET_ZIP}. "
            "Set DATASET_ROOT or DATASET_ZIP."
        )

    DATASET_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    marker = DATASET_EXTRACT_DIR / "metadata" / "document_registry_final.csv"
    nested_extract = DATASET_EXTRACT_DIR / "foxconn synthetic data"
    nested_marker = nested_extract / "metadata" / "document_registry_final.csv"

    if not marker.exists() and not nested_marker.exists():
        with zipfile.ZipFile(DATASET_ZIP) as archive:
            archive.extractall(DATASET_EXTRACT_DIR)

    if marker.exists():
        return DATASET_EXTRACT_DIR
    if nested_marker.exists():
        return nested_extract

    raise FileNotFoundError(
        f"Extracted zip at {DATASET_EXTRACT_DIR} does not contain the expected metadata folder."
    )
