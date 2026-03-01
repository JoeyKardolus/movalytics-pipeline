"""Shared signal filtering utilities for temporal smoothing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, medfilt


def butterworth_lowpass(
    data: np.ndarray,
    cutoff_hz: float,
    fps: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth low-pass filter applied per column.

    Handles short signals (< 3*padlen) and flat signals gracefully.
    Returns a copy; input is not modified.

    Args:
        data: (N,) or (N, D) array — each column filtered independently.
        cutoff_hz: Cutoff frequency in Hz.
        fps: Sampling rate in Hz.
        order: Filter order (default 4).

    Returns:
        Filtered array with same shape as input.
    """
    nyq = fps / 2.0
    if cutoff_hz >= nyq or cutoff_hz <= 0:
        return data.copy()

    b, a = butter(order, cutoff_hz / nyq, btype="low")
    result = data.copy()

    if result.ndim == 1:
        if np.std(result) > 1e-6:
            try:
                result = filtfilt(b, a, result)
            except ValueError:
                pass  # Signal too short for filtfilt
    else:
        for col in range(result.shape[1]):
            signal = result[:, col]
            if np.std(signal) > 1e-6:
                try:
                    result[:, col] = filtfilt(b, a, signal)
                except ValueError:
                    pass
    return result


def median_filter_1d(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """1D median filter applied per column.

    Wraps scipy.signal.medfilt. Returns a copy.

    Args:
        data: (N,) or (N, D) array.
        kernel_size: Window size (must be odd).

    Returns:
        Filtered array with same shape as input.
    """
    result = data.copy()
    if result.ndim == 1:
        result = medfilt(result, kernel_size=kernel_size)
    else:
        for col in range(result.shape[1]):
            result[:, col] = medfilt(result[:, col], kernel_size=kernel_size)
    return result


def moving_average(
    data: np.ndarray,
    window: int,
    mode: str = "reflect",
) -> np.ndarray:
    """Uniform moving average applied per column.

    Wraps scipy.ndimage.uniform_filter1d. Returns a copy.

    Args:
        data: (N,) or (N, D) array.
        window: Smoothing window size.
        mode: Edge handling mode (default 'reflect').

    Returns:
        Smoothed array with same shape as input.
    """
    result = data.copy()
    if result.ndim == 1:
        result = uniform_filter1d(result, size=window, mode=mode)
    else:
        for col in range(result.shape[1]):
            result[:, col] = uniform_filter1d(
                result[:, col], size=window, mode=mode
            )
    return result
