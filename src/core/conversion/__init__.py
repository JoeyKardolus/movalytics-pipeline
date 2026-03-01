"""Conversion module for TRC file I/O and marker mapping."""

from .trc_io import (
    load_trc,
    save_trc,
    read_trc_raw,
    write_trc_raw,
    build_trc_header,
    MARKER_NAMES_21,
)

__all__ = [
    "load_trc",
    "save_trc",
    "read_trc_raw",
    "write_trc_raw",
    "build_trc_header",
    "MARKER_NAMES_21",
]
