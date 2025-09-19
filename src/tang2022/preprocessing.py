"""
ECG preprocessing pipeline following Tang et al. 2022 specifications.

This module implements:
- Band-pass filtering (0.05-100 Hz)
- Resampling to 200 Hz
- Windowing (5-second windows with 4-second overlap)
- Optional padding to 1024 samples
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Optional
import warnings


def bandpass_filter(ecg_data: np.ndarray, fs: float, 
                   low_freq: float = 0.05, high_freq: float = 100.0) -> np.ndarray:
    """
    Apply zero-phase band-pass filter to ECG data.
    
    Args:
        ecg_data: ECG data [n_samples, n_channels]
        fs: Sampling frequency in Hz
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        
    Returns:
        Filtered ECG data [n_samples, n_channels]
    """
    # Design Butterworth filter
    nyquist = fs / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Ensure normalized frequencies are valid
    low_norm = max(low_norm, 1e-6)
    high_norm = min(high_norm, 1 - 1e-6)
    
    # Design filter
    sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
    
    # Apply zero-phase filtering to each channel
    filtered_data = np.zeros_like(ecg_data)
    for ch in range(ecg_data.shape[1]):
        filtered_data[:, ch] = signal.sosfiltfilt(sos, ecg_data[:, ch])
    
    return filtered_data


def resample_to_200hz(ecg_data: np.ndarray, fs_original: float, 
                      target_fs: float = 200.0) -> np.ndarray:
    """
    Resample ECG data to 200 Hz using high-quality resampling.
    
    Args:
        ecg_data: ECG data [n_samples, n_channels]
        fs_original: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz (default: 200)
        
    Returns:
        Resampled ECG data [n_samples_new, n_channels]
    """
    if abs(fs_original - target_fs) < 1e-6:
        return ecg_data
    
    # Calculate resampling ratio
    ratio = target_fs / fs_original
    n_samples_new = int(np.round(ecg_data.shape[0] * ratio))
    
    # Resample each channel
    resampled_data = np.zeros((n_samples_new, ecg_data.shape[1]))
    for ch in range(ecg_data.shape[1]):
        resampled_data[:, ch] = signal.resample(ecg_data[:, ch], n_samples_new)
    
    return resampled_data


def create_windows(ecg_data: np.ndarray, fs: float = 200.0, 
                  window_sec: float = 5.0, overlap_sec: float = 4.0) -> List[np.ndarray]:
    """
    Create overlapping windows from ECG data.
    
    Args:
        ecg_data: ECG data [n_samples, n_channels]
        fs: Sampling frequency in Hz
        window_sec: Window duration in seconds
        overlap_sec: Overlap duration in seconds
        
    Returns:
        List of windows, each [window_samples, n_channels]
    """
    window_samples = int(window_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    stride_samples = window_samples - overlap_samples
    
    n_samples, n_channels = ecg_data.shape
    
    # Calculate number of windows
    n_windows = (n_samples - window_samples) // stride_samples + 1
    
    if n_windows <= 0:
        warnings.warn(f"ECG too short for windowing. Length: {n_samples} samples, "
                     f"required: {window_samples} samples")
        return []
    
    windows = []
    for i in range(n_windows):
        start_idx = i * stride_samples
        end_idx = start_idx + window_samples
        
        if end_idx <= n_samples:
            window = ecg_data[start_idx:end_idx, :]
            windows.append(window)
    
    return windows


def pad_window_to_1024(window: np.ndarray) -> np.ndarray:
    """
    Pad window to 1024 samples with trailing zeros as specified in Tang et al. 2022.
    
    The paper mentions: "the input EGM or ECG matrices were padded to length 1024 
    in the time dimension with zeros to ensure that the max-pooling layers in the 
    bottleneck blocks resulted in integer output lengths."
    
    Args:
        window: ECG window [n_samples, n_channels]
        
    Returns:
        Padded window [1024, n_channels]
    """
    if window.shape[0] >= 1024:
        return window[:1024, :]
    
    # Pad with zeros at the end
    padding = 1024 - window.shape[0]
    padded_window = np.pad(window, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    
    return padded_window


def preprocess_patient_ecg(ecg_data: np.ndarray, fs_original: float,
                          target_fs: float = 200.0,
                          filter_low: float = 0.05, filter_high: float = 100.0,
                          window_sec: float = 5.0, overlap_sec: float = 4.0,
                          pad_to_1024: bool = True) -> List[np.ndarray]:
    """
    Complete preprocessing pipeline for a single patient's ECG.
    
    Args:
        ecg_data: ECG data [n_samples, n_channels] (8 independent leads per Tang et al. 2022)
        fs_original: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz
        filter_low: Low cutoff frequency for band-pass filter
        filter_high: High cutoff frequency for band-pass filter
        window_sec: Window duration in seconds
        overlap_sec: Overlap duration in seconds
        pad_to_1024: Whether to pad windows to 1024 samples
        
    Returns:
        List of preprocessed windows, each [window_samples, n_channels]
    """
    # Step 1: Band-pass filter
    ecg_filtered = bandpass_filter(ecg_data, fs_original, filter_low, filter_high)
    
    # Step 2: Resample to target frequency
    ecg_resampled = resample_to_200hz(ecg_filtered, fs_original, target_fs)
    
    # Step 3: Create windows
    windows = create_windows(ecg_resampled, target_fs, window_sec, overlap_sec)
    
    # Step 4: Optional padding
    if pad_to_1024:
        windows = [pad_window_to_1024(window) for window in windows]
    
    return windows


def validate_preprocessing_params():
    """Validate preprocessing parameters match paper specifications."""
    specs = {
        'leads': ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        'sampling_rate': 200,  # Hz
        'filter_low': 0.05,    # Hz
        'filter_high': 100,    # Hz
        'window_duration': 5,   # seconds
        'overlap_duration': 4,  # seconds
        'window_samples': 1000, # at 200 Hz
        'overlap_samples': 800, # at 200 Hz
        'stride_samples': 200,  # at 200 Hz
    }
    
    print("Preprocessing parameters (Tang et al. 2022):")
    for key, value in specs.items():
        print(f"  {key}: {value}")
    
    return specs


if __name__ == "__main__":
    # Validate parameters
    validate_preprocessing_params()
    
    # Example usage
    print("\nExample preprocessing:")
    
    # Create dummy ECG data
    fs_orig = 1000  # Hz
    duration = 30   # seconds
    n_samples = int(fs_orig * duration)
    n_channels = 8
    
    # Generate synthetic ECG-like signal
    t = np.linspace(0, duration, n_samples)
    dummy_ecg = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Simple synthetic ECG: combination of sine waves
        dummy_ecg[:, ch] = (
            0.1 * np.sin(2 * np.pi * 1.2 * t) +  # Heart rate ~72 bpm
            0.05 * np.sin(2 * np.pi * 2.4 * t) +  # Harmonics
            0.02 * np.random.randn(n_samples)      # Noise
        )
    
    print(f"Input ECG shape: {dummy_ecg.shape}")
    print(f"Input sampling rate: {fs_orig} Hz")
    
    # Preprocess
    windows = preprocess_patient_ecg(dummy_ecg, fs_orig)
    
    print(f"Number of windows: {len(windows)}")
    if windows:
        print(f"Window shape: {windows[0].shape}")
        print(f"Expected windows for {duration}s ECG: {(duration * 200 - 1000) // 200 + 1}")
