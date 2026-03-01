#!/usr/bin/env python3
"""Visualize MeTRAbs TRC output as animated 3D skeleton.

Usage:
    uv run python scripts/viz/trc_3d_viewer.py data/output/joey/joey_metrabs.trc
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.conversion.trc_io import load_trc

# Skeleton connections for visualization (pairs of marker name substrings)
SKELETON_LINKS = [
    ("pelv", "lhip"), ("pelv", "rhip"), ("pelv", "backneck"),
    ("backneck", "lshom"), ("backneck", "rshom"),
    ("lhip", "lkne"), ("rhip", "rkne"),
    ("lkne", "lank"), ("rkne", "rank"),
    ("lshom", "lelb"), ("rshom", "relb"),
    ("lelb", "lwri"), ("relb", "rwri"),
    ("lank", "ltoe"), ("rank", "rtoe"),
    ("lank", "lhee"), ("rank", "rhee"),
]


def find_marker_idx(names: list[str], name: str) -> int | None:
    """Find marker index by exact case-insensitive match."""
    lower = [n.lower() for n in names]
    target = name.lower()
    if target in lower:
        return lower.index(target)
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python trc_3d_viewer.py <trc_file>")
        sys.exit(1)

    trc_path = Path(sys.argv[1])
    data, marker_names, fps = load_trc(trc_path)
    n_frames = data.shape[0]
    print(f"[trc_viewer] {n_frames} frames, {len(marker_names)} markers, {fps:.1f} fps")

    # TRC coords: X=forward, Y=up, Z=right
    # Plot as: X=forward (horizontal), Z=right (horizontal), Y=up (vertical)
    x_all = data[:, :, 0]  # forward
    y_all = data[:, :, 1]  # up
    z_all = data[:, :, 2]  # right

    # Build skeleton link indices
    links = []
    for a_sub, b_sub in SKELETON_LINKS:
        a_idx = find_marker_idx(marker_names, a_sub)
        b_idx = find_marker_idx(marker_names, b_sub)
        if a_idx is not None and b_idx is not None:
            links.append((a_idx, b_idx))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Set fixed limits from data range
    margin = 0.3
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    z_min, z_max = np.nanmin(z_all), np.nanmax(z_all)

    scatter = ax.scatter([], [], [], c="blue", s=8)
    line_objs = [ax.plot([], [], [], "b-", linewidth=1.5, alpha=0.7)[0] for _ in links]
    title = ax.set_title("")

    def update(frame_idx):
        fx = x_all[frame_idx]
        fy = y_all[frame_idx]
        fz = z_all[frame_idx]

        # Plot: matplotlib 3D axes as (X_plot=forward, Y_plot=right, Z_plot=up)
        scatter._offsets3d = (fx, fz, fy)

        for i, (a, b) in enumerate(links):
            line_objs[i].set_data_3d(
                [fx[a], fx[b]], [fz[a], fz[b]], [fy[a], fy[b]]
            )

        # Follow the skeleton center
        cx = np.nanmean(fx)
        cz = np.nanmean(fz)
        ax.set_xlim(cx - 1.5, cx + 1.5)
        ax.set_ylim(cz - 1.5, cz + 1.5)
        ax.set_zlim(y_min - margin, y_max + margin)

        title.set_text(f"MeTRAbs TRC — Frame {frame_idx}/{n_frames} ({frame_idx/fps:.2f}s)")
        return [scatter] + line_objs + [title]

    ax.set_xlabel("Forward (X)")
    ax.set_ylabel("Right (Z)")
    ax.set_zlabel("Up (Y)")

    interval = max(1, int(1000 / fps))
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
