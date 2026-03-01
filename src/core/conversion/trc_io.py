"""TRC file I/O module.

Read/write functions for TRC (Track Row Column) motion capture files
in OpenSim-compatible format.

TRC Format:
    Line 0: PathFileType	4	(X/Y/Z)	filename.trc
    Line 1: DataRate	30.00	CameraRate	30.00	NumFrames	100	NumMarkers	21	Units	m
    Line 2: (empty)
    Line 3: Frame#	Time	Marker1	Marker1	Marker1	Marker2	...
    Line 4: 		X1	Y1	Z1	X2	...
    Lines 5+: 1	0.000000	x	y	z	x	...
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.shared.constants import MARKER_NAMES_21  # re-exported for backwards compat


def load_trc(trc_path: Path | str) -> Tuple[np.ndarray, List[str], float]:
    """Load TRC file to numpy array.

    Uses actual data column count as authoritative.

    Args:
        trc_path: Path to TRC file

    Returns:
        data: (n_frames, n_markers, 3) marker positions as float32
        marker_names: List of marker names
        frame_rate: Data rate from header
    """
    trc_path = Path(trc_path)
    lines = trc_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if len(lines) < 5:
        raise ValueError(f"TRC file too short: {trc_path}")

    # Parse header (Line 1: DataRate info, Line 2: values in standard format)
    frame_rate = _parse_frame_rate(lines[1], lines[2] if len(lines) > 2 else "")

    # Parse marker names from Line 3 (standard) or Line 3 (interleaved, where line 2 is empty)
    # Find first line with "Frame#" to locate marker names
    marker_line_idx = 3
    for i in range(2, min(5, len(lines))):
        if lines[i].strip().startswith("Frame#"):
            marker_line_idx = i
            break
    header_marker_names = _parse_marker_names(lines[marker_line_idx])

    # Find data start (first line with numeric frame number)
    data_start = _find_data_start(lines)

    # Parse data rows
    data_rows = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if not _is_data_row(parts):
            continue

        coords = []
        for val in parts[2:]:  # Skip Frame# and Time
            try:
                coords.append(float(val) if val.strip() else np.nan)
            except ValueError:
                coords.append(np.nan)
        data_rows.append(coords)

    if not data_rows:
        raise ValueError(f"No data rows found in {trc_path}")

    # Convert to array
    data = np.array(data_rows, dtype=np.float32)
    n_frames = data.shape[0]
    n_data_cols = data.shape[1]
    n_markers = n_data_cols // 3

    # Determine marker names
    marker_names = _determine_marker_names(n_markers, header_marker_names)

    # Pad/trim data to match marker count
    expected_cols = n_markers * 3
    if n_data_cols < expected_cols:
        padding = np.full((n_frames, expected_cols - n_data_cols), np.nan, dtype=np.float32)
        data = np.hstack([data, padding])
    elif n_data_cols > expected_cols:
        data = data[:, :expected_cols]

    data = data.reshape(n_frames, n_markers, 3)
    return data, marker_names, frame_rate


def save_trc(
    data: np.ndarray,
    marker_names: List[str],
    output_path: Path | str,
    frame_rate: float = 30.0,
) -> None:
    """Write marker data to TRC file.

    Args:
        data: (n_frames, n_markers, 3) marker positions
        marker_names: List of marker names (must match data.shape[1])
        output_path: Output TRC file path
        frame_rate: Data frame rate in Hz
    """
    output_path = Path(output_path)
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Data must be shape (n_frames, n_markers, 3), got {data.shape}")

    n_frames, n_markers, _ = data.shape

    if len(marker_names) != n_markers:
        raise ValueError(f"marker_names length {len(marker_names)} != data n_markers {n_markers}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Line 0: PathFileType
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}\n")

        # Line 1: Key names (standard TRC: 8 keys on line 1, 8 values on line 2)
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
                "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")

        # Line 2: Values
        f.write(f"{frame_rate:.2f}\t{frame_rate:.2f}\t{n_frames}\t{n_markers}\t"
                f"mm\t{frame_rate:.2f}\t1\t{n_frames}\n")

        # Line 3: Marker names (each once, followed by 2 empty tabs for Y/Z columns)
        name_parts = ["Frame#", "Time"]
        for name in marker_names:
            name_parts.extend([name, "", ""])
        f.write("\t".join(name_parts) + "\n")

        # Line 4: X Y Z labels
        xyz_parts = ["", ""]
        for i in range(n_markers):
            xyz_parts.extend([f"X{i + 1}", f"Y{i + 1}", f"Z{i + 1}"])
        f.write("\t".join(xyz_parts) + "\n")

        # Data rows
        for frame_idx in range(n_frames):
            time = frame_idx / frame_rate
            row_parts = [str(frame_idx + 1), f"{time:.6f}"]
            for marker_idx in range(n_markers):
                for coord_idx in range(3):
                    val = data[frame_idx, marker_idx, coord_idx]
                    if np.isnan(val):
                        row_parts.append("")
                    else:
                        row_parts.append(f"{val:.6f}")
            f.write("\t".join(row_parts) + "\n")


def read_trc_raw(trc_path: Path | str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read TRC file preserving header and raw data.

    Lower-level function for operations that need access to header lines
    or Frame#/Time columns directly.

    Args:
        trc_path: Path to TRC file

    Returns:
        header_lines: First 5 lines of file
        frames: Frame numbers as int array
        times: Timestamps as float array
        coords: (n_frames, n_markers, 3) coordinates
    """
    trc_path = Path(trc_path)
    lines = trc_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if len(lines) < 5:
        raise ValueError(f"TRC file too short: {trc_path}")

    header_lines = lines[:5]
    data_lines = [line for line in lines[5:] if line.strip()]

    if not data_lines:
        raise ValueError(f"No data rows found in {trc_path}")

    # Determine marker count from first data line
    first_row = data_lines[0].split("\t")
    num_data_cols = len(first_row) - 2
    num_markers = num_data_cols // 3

    num_frames = len(data_lines)
    frames = np.zeros(num_frames, dtype=int)
    times = np.zeros(num_frames, dtype=float)
    coords = np.full((num_frames, num_markers, 3), np.nan, dtype=float)

    for fi, line in enumerate(data_lines):
        cols = line.split("\t")
        try:
            frames[fi] = int(float(cols[0]))
            times[fi] = float(cols[1])
        except (ValueError, IndexError):
            continue

        for mi in range(num_markers):
            cx, cy, cz = 2 + 3 * mi, 3 + 3 * mi, 4 + 3 * mi
            if cz < len(cols):
                try:
                    coords[fi, mi, 0] = float(cols[cx]) if cols[cx] else np.nan
                    coords[fi, mi, 1] = float(cols[cy]) if cols[cy] else np.nan
                    coords[fi, mi, 2] = float(cols[cz]) if cols[cz] else np.nan
                except ValueError:
                    pass

    return header_lines, frames, times, coords


def write_trc_raw(
    output_path: Path | str,
    header_lines: List[str],
    frames: np.ndarray,
    times: np.ndarray,
    coords: np.ndarray,
) -> None:
    """Write TRC file from raw components.

    Lower-level function for operations that need control over header lines
    or Frame#/Time columns directly.

    Args:
        output_path: Output file path
        header_lines: First 5 lines of TRC header
        frames: Frame numbers as int array
        times: Timestamps as float array
        coords: (n_frames, n_markers, 3) coordinates
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames, num_markers, _ = coords.shape
    data_lines = []

    for fi in range(num_frames):
        row = [str(frames[fi]), f"{times[fi]:.6f}"]
        for mi in range(num_markers):
            x, y, z = coords[fi, mi]
            if np.isnan(x):
                row.extend(["", "", ""])
            else:
                row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        data_lines.append("\t".join(row))

    output_content = "\n".join(header_lines + data_lines) + "\n"
    output_path.write_text(output_content, encoding="utf-8")


def build_trc_header(
    filename: str,
    n_frames: int,
    n_markers: int,
    marker_names: List[str] | None = None,
    frame_rate: float = 30.0,
) -> List[str]:
    """Build TRC header lines.

    Args:
        filename: TRC filename for header
        n_frames: Number of frames
        n_markers: Number of markers
        marker_names: List of marker names (auto-generated if None)
        frame_rate: Data rate in Hz

    Returns:
        List of 5 header lines
    """
    if marker_names is None:
        if n_markers == 21:
            marker_names = MARKER_NAMES_21
        else:
            marker_names = [f"Marker{i + 1}" for i in range(n_markers)]
    elif len(marker_names) != n_markers:
        raise ValueError(f"marker_names length {len(marker_names)} != n_markers {n_markers}")

    # Line 0: PathFileType
    line0 = f"PathFileType\t4\t(X/Y/Z)\t{filename}\n"

    # Line 1: Key names (standard TRC format)
    line1 = ("DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
             "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")

    # Line 2: Values
    line2 = (f"{frame_rate:.2f}\t{frame_rate:.2f}\t{n_frames}\t{n_markers}\t"
             f"mm\t{frame_rate:.2f}\t1\t{n_frames}\n")

    # Line 3: Marker names (each once, followed by 2 empty tabs for Y/Z columns)
    marker_cols = "\t".join(f"{m}\t\t" for m in marker_names)
    line3 = f"Frame#\tTime\t{marker_cols}\n"

    # Line 4: X Y Z labels
    xyz_labels = "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(n_markers))
    line4 = f"\t\t{xyz_labels}\n"

    return [line0, line1, line2, line3, line4]


# --- Private helpers ---

def _parse_frame_rate(header_line: str, values_line: str = "") -> float:
    """Parse frame rate from TRC header.

    Handles two formats:
    - Standard TRC: line 1 has key names, line 2 has values
    - Legacy interleaved: line 1 has key-value pairs (DataRate\t30.00\t...)
    """
    parts = header_line.strip().split("\t")
    # Try interleaved format first: DataRate <value> CameraRate <value> ...
    for i, part in enumerate(parts):
        if part == "DataRate" and i + 1 < len(parts):
            try:
                return float(parts[i + 1])
            except ValueError:
                pass
    # Standard format: line 1 has keys, line 2 has values
    if values_line:
        vals = values_line.strip().split("\t")
        if vals:
            try:
                return float(vals[0])  # First value = DataRate
            except ValueError:
                pass
    return 30.0  # Default fallback


def _parse_marker_names(marker_line: str) -> List[str]:
    """Parse unique marker names from header line 3."""
    parts = marker_line.strip().split("\t")
    names = []
    for i in range(2, len(parts), 3):  # Skip Frame#, Time; step by 3
        if parts[i] and parts[i] not in names:
            names.append(parts[i])
    return names


def _find_data_start(lines: List[str]) -> int:
    """Find the line index where data rows begin.

    Skips header lines (0-4 in standard TRC). Looks for first numeric
    row after the XYZ labels line.
    """
    # Start after minimum header (line 0: PathFileType, line 1-2: metadata,
    # line 3: marker names, line 4: XYZ labels)
    for i, line in enumerate(lines):
        if i < 4:
            continue
        parts = line.strip().split("\t")
        if _is_data_row(parts):
            return i
    return 5  # Default TRC data start


def _is_data_row(parts: List[str]) -> bool:
    """Check if parts represent a data row (starts with numeric frame)."""
    if not parts or not parts[0]:
        return False
    # Check if first column is numeric (frame number)
    test = parts[0].replace(".", "", 1).replace("-", "", 1)
    return test.isdigit()


def _determine_marker_names(n_markers: int, header_names: List[str]) -> List[str]:
    """Determine marker names based on count and header."""
    if n_markers <= len(header_names):
        return header_names[:n_markers]
    else:
        # Generate names for missing markers
        names = header_names.copy()
        for i in range(len(header_names), n_markers):
            names.append(f"Marker{i + 1}")
        return names
