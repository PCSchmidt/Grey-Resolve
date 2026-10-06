"""Fetch pretrained backbone weights at runtime (never committed to git).

Downloads the InsightFace buffalo_l model pack and extracts only the two ONNX
models Grey-Resolve uses:
- det_10g.onnx    -- SCRFD face detector (with 5-point landmarks)
- w600k_r50.onnx  -- ArcFace embedding model (512-d)

The pack also contains landmark/genderage models we deliberately do not use.

Weights are non-commercial (see THIRD_PARTY_NOTICES.md and
docs/BACKBONE_LICENSES.md) and must not be redistributed in this repository.

Usage:
    python scripts/fetch_backbone.py [--root ~/.insightface] [--pack buffalo_l] [--force]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
DEFAULT_PACK = "buffalo_l"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.home() / ".insightface"), help="insightface model root")
    parser.add_argument("--pack", default=DEFAULT_PACK, help="model pack name (zip on the insightface release)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    from insightface.utils import storage

    dest_dir = Path(args.root) / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)

    missing = [m for m in MODEL_FILES if args.force or not (dest_dir / m).exists()]
    if not missing:
        print(f"all models already present in {dest_dir}")
        return

    zip_path = Path(args.root) / "models" / f"{args.pack}.zip"
    url = storage.model_zoo_download_url(f"{args.pack}.zip")
    print(f"downloading {url}")
    storage.download_file(url, path=str(zip_path), overwrite=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for model_file in missing:
            matches = [n for n in names if n.endswith(model_file)]
            if not matches:
                raise FileNotFoundError(f"{model_file} not found inside {zip_path.name}")
            target = dest_dir / model_file
            with zf.open(matches[0]) as src, open(target, "wb") as out:
                out.write(src.read())
            print(f"extracted: {target}")
    zip_path.unlink(missing_ok=True)
    print("remember: these weights are non-commercial; do not commit them.")


if __name__ == "__main__":
    main()
