"""Verify that ACIS data versioning files and pipeline outputs are present."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_RAW_DATA = Path("data/raw/insurance_data.csv")
DEFAULT_RAW_DVC = Path("data/raw/insurance_data.csv.dvc")
DEFAULT_CLEANED_DATA = Path("data/processed/insurance_data_cleaned.csv")
DEFAULT_DVC_YAML = Path("dvc.yaml")
DEFAULT_DVC_LOCK = Path("dvc.lock")


def check_path(path: Path, description: str) -> bool:
    """Print whether an expected file exists."""
    exists = path.exists()
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {description}: {path}")
    return exists


def resolve_dvc_command() -> str | None:
    """Return the available DVC command, if DVC is installed."""
    dvc_command = shutil.which("dvc")
    if dvc_command:
        return dvc_command

    local_dvc = Path(".venv/Scripts/dvc.exe")
    if local_dvc.exists():
        return str(local_dvc)

    return None


def run_dvc_status() -> bool:
    """Run `dvc status` to confirm whether tracked data is up to date."""
    dvc_command = resolve_dvc_command()
    if not dvc_command:
        print("[MISSING] DVC executable: install dependencies or activate .venv")
        return False

    result = subprocess.run(
        [dvc_command, "status"],
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip() or result.stderr.strip()
    status = "OK" if result.returncode == 0 else "ERROR"
    print(f"[{status}] dvc status")
    if output:
        print(output)

    return result.returncode == 0


def verify_dvc_setup(
    raw_data: Path = DEFAULT_RAW_DATA,
    raw_dvc: Path = DEFAULT_RAW_DVC,
    cleaned_data: Path = DEFAULT_CLEANED_DATA,
) -> bool:
    """Verify raw data, DVC metadata, cleaned output, and pipeline status."""
    checks = [
        check_path(raw_dvc, "Raw dataset DVC metadata"),
        check_path(raw_data, "Raw dataset restored locally"),
        check_path(cleaned_data, "Cleaned dataset output"),
        check_path(DEFAULT_DVC_YAML, "DVC pipeline definition"),
        check_path(DEFAULT_DVC_LOCK, "DVC pipeline lock file"),
        run_dvc_status(),
    ]

    if all(checks):
        print("\nDVC verification passed. Raw and cleaned data are reproducible.")
        return True

    print("\nDVC verification found missing files or an out-of-date pipeline.")
    return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DVC verification command."""
    parser = argparse.ArgumentParser(description="Verify ACIS DVC setup.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=DEFAULT_RAW_DATA,
        help="Path to the restored raw dataset.",
    )
    parser.add_argument(
        "--raw-dvc",
        type=Path,
        default=DEFAULT_RAW_DVC,
        help="Path to the raw dataset .dvc metadata file.",
    )
    parser.add_argument(
        "--cleaned-data",
        type=Path,
        default=DEFAULT_CLEANED_DATA,
        help="Path to the cleaned pipeline output.",
    )
    return parser.parse_args()


def main() -> None:
    """Run DVC verification and exit with a non-zero status on failure."""
    args = parse_args()
    passed = verify_dvc_setup(args.raw_data, args.raw_dvc, args.cleaned_data)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
