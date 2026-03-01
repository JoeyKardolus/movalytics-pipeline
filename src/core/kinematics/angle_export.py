"""Joint angle CSV export for clinical DOF data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_comprehensive_angles_csv(
    angle_results: dict[str, pd.DataFrame],
    output_dir: Path,
    basename: str,
) -> None:
    """Save all joint angle data to separate CSV files.

    Args:
        angle_results: Dict of DataFrames keyed by joint group name
        output_dir: Directory to save CSV files
        basename: Base name for files (e.g., "joey")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for joint_name, df in angle_results.items():
        csv_path = output_dir / f"{basename}_angles_{joint_name}.csv"
        df.to_csv(csv_path, index=False, float_format="%.3f")
        print(f"[save_angles] {csv_path.name}")

    print(f"[save_angles] Saved {len(angle_results)} CSV files to {output_dir}")
