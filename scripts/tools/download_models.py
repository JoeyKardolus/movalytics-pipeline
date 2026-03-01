#!/usr/bin/env python3
"""Download pretrained models for the pipeline.

SAM 3D Body weights are hosted on Google Drive and must be downloaded before
running the pipeline. This script handles downloading + verification.

YOLOX models are auto-downloaded by rtmlib on first use.

Usage:
    uv run python scripts/tools/download_models.py              # download all
    uv run python scripts/tools/download_models.py --check      # check status only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SAM3D_DIR = PROJECT_ROOT / "models" / "sam-3d-body-dinov3"
SAM3D_ASSETS_DIR = SAM3D_DIR / "assets"

# Google Drive file IDs
MODELS = {
    "sam3d_checkpoint": {
        "path": SAM3D_DIR / "model.ckpt",
        "gdrive_id": "1Gx5LSKwLIJB2ctnAMrA2FHZdn91hsySS",
        "size_mb": 2000,
        "description": "SAM 3D Body checkpoint (DINOv3 backbone)",
    },
    "mhr_model": {
        "path": SAM3D_ASSETS_DIR / "mhr_model.pt",
        "gdrive_id": "1nNsGax1Ni5Gi9nl2gErHrdxLfLQUqqUk",
        "size_mb": 664,
        "description": "MHR body model",
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _ensure_gdown() -> bool:
    """Ensure gdown is installed."""
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        print("  Installing gdown (Google Drive downloader)...")
        subprocess.check_call(["uv", "pip", "install", "gdown", "-q"])
        return True


def _download_file(gdrive_id: str, dest: Path, description: str) -> bool:
    """Download a file from Google Drive with gdown."""
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"  Downloading {description}...")
    print(f"    Destination: {dest}")

    try:
        gdown.download(url, str(dest), quiet=False)
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"    Done: {size_mb:.0f} MB")
            return True
        else:
            print(f"    FAILED: file not created")
            return False
    except Exception as e:
        print(f"\n    FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Check / download
# ---------------------------------------------------------------------------

def check_models() -> bool:
    """Check that all required models are available."""
    all_ok = True

    print("\n=== Model Status ===\n")

    for key, info in MODELS.items():
        path = info["path"]
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            expected = info["size_mb"]
            if size_mb < expected * 0.9:
                print(f"  {info['description']}: WARNING — file too small "
                      f"({size_mb:.0f} MB, expected ~{expected} MB)")
                all_ok = False
            else:
                print(f"  {info['description']}: OK ({size_mb:.0f} MB)")
        else:
            print(f"  {info['description']}: MISSING")
            print(f"    Expected at: {path}")
            all_ok = False

    # YOLOX (auto-downloaded by rtmlib)
    print(f"  YOLOX: auto-downloaded by rtmlib on first use")

    print()
    if all_ok:
        print("All required models available. Ready to run the pipeline.")
    else:
        print("Some models are missing. Run without --check to download.")

    return all_ok


def download_models() -> bool:
    """Download any missing models from Google Drive."""
    _ensure_gdown()

    all_ok = True

    for key, info in MODELS.items():
        path = info["path"]
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb >= info["size_mb"] * 0.9:
                print(f"  {info['description']}: already downloaded ({size_mb:.0f} MB)")
                continue

        if not _download_file(info["gdrive_id"], path, info["description"]):
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify pipeline model weights."
    )
    parser.add_argument("--check", action="store_true",
                        help="Only check status, don't download")
    args = parser.parse_args()

    if args.check:
        ok = check_models()
    else:
        print("=== Downloading missing models ===\n")
        ok = download_models()
        print()
        check_models()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
