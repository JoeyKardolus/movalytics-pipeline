"""OpenSim IK subprocess launcher.

Runs src/workers/opensim_ik_worker.py in the opensim conda env
to perform scaling + inverse kinematics on TRC marker data.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


_WORKER_SCRIPT = Path(__file__).parent.parent.parent / "workers" / "opensim_ik_worker.py"


def run_opensim_ik(
    trc_path: Path | str,
    output_dir: Path | str,
    subject_height: float = 1.75,
    subject_mass: float = 70.0,
    conda_env: str = "opensim",
    timeout: int = 300,
    sam3d_npz: Path | str | None = None,
    skip_fk: bool = False,
) -> Path:
    """Run OpenSim IK in the opensim conda env.

    Args:
        trc_path: Path to TRC file with marker positions (mm).
        output_dir: Directory for output files (.mot, .osim).
        subject_height: Subject height in meters.
        subject_mass: Subject mass in kg.
        conda_env: Conda environment name with OpenSim.
        timeout: Maximum time in seconds for IK to complete.
        sam3d_npz: Path to SAM 3D NPZ with rest_joint_coords for per-segment scaling.

    Returns:
        Path to output .mot file.

    Raises:
        RuntimeError: If IK fails or .mot not found in output.
    """
    trc_path = Path(trc_path).resolve()
    output_dir = Path(output_dir).resolve()
    worker = _WORKER_SCRIPT.resolve()

    if not worker.exists():
        raise FileNotFoundError(f"OpenSim IK worker not found: {worker}")
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    # Use full path to conda env's Python to avoid PATH conflicts
    # (uv run puts .venv/bin first, which conda run inherits)
    conda_bin = shutil.which("conda")
    if conda_bin:
        conda_prefix = Path(conda_bin).parent.parent
        conda_python = conda_prefix / "envs" / conda_env / "bin" / "python"
    else:
        conda_python = Path("python")
    if not conda_python.exists():
        conda_python = Path("python")

    cmd = [
        "conda", "run", "-n", conda_env,
        str(conda_python), str(worker),
        "--trc", str(trc_path),
        "--output-dir", str(output_dir),
        "--height", str(subject_height),
        "--mass", str(subject_mass),
    ]
    if sam3d_npz is not None:
        cmd.extend(["--npz", str(Path(sam3d_npz).resolve())])
    if skip_fk:
        cmd.append("--skip-fk")

    print(f"[opensim-ik] Running: conda run -n {conda_env} python opensim_ik_worker.py")

    timed_out = False
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Print worker output
        for line in result.stdout.splitlines():
            if line.startswith("[opensim-ik]"):
                print(line)

        if result.returncode != 0:
            print(f"[opensim-ik] STDERR:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(
                f"OpenSim IK failed (exit code {result.returncode}): "
                f"{result.stderr[:500]}"
            )

        # Parse .mot path from stdout
        mot_path = None
        for line in result.stdout.splitlines():
            if line.startswith("MOT_PATH="):
                mot_path = Path(line.split("=", 1)[1].strip())
                break
    except subprocess.TimeoutExpired:
        timed_out = True
        mot_path = None

    if mot_path is None or not mot_path.exists():
        # Fallback: look for .mot file in output dir (covers timeout case
        # where IK completed but FK or conda run hung)
        mot_files = list(output_dir.glob("*_ik.mot"))
        if mot_files:
            mot_path = mot_files[0]
            if timed_out:
                print(f"[opensim-ik] Timeout after {timeout}s but .mot exists — continuing")
        else:
            raise RuntimeError(
                f"OpenSim IK {'timed out' if timed_out else 'completed'} "
                f"but no .mot file found in {output_dir}"
            )

    print(f"[opensim-ik] Output: {mot_path}")
    return mot_path
