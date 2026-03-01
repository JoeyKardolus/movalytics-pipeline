"""Output directory cleanup utilities."""

from pathlib import Path


def cleanup_output_directory(run_dir: Path, video_stem: str) -> None:
    """Organize output directory after pipeline completion.

    Actions:
    - Remove WSL Zone.Identifier metadata files
    - Organize joint angle files into joint_angles/ subdirectory
    """
    # Remove Zone.Identifier files (WSL metadata)
    for f in run_dir.glob("*.Zone.Identifier"):
        f.unlink(missing_ok=True)

    # Organize joint angle files
    angle_dir = run_dir / "joint_angles"
    angle_files = (
        list(run_dir.glob(f"{video_stem}_angles_*.csv"))
        + list(run_dir.glob(f"{video_stem}_all_joint_angles.png"))
        + list(run_dir.glob(f"{video_stem}_joint_angles_comparison.png"))
    )

    if angle_files:
        angle_dir.mkdir(exist_ok=True)
        for f in angle_files:
            dest = angle_dir / f.name
            dest.unlink(missing_ok=True)
            f.rename(dest)
