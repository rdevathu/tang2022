"""
Data augmentation transforms for ECG windows following Tang et al. 2022 specifications.

Implements the five augmentations described in the paper:
1. Random time shift within the 5-sec window by up to ±2.5 sec
2. Random amplitude scaling: multiply signal by a factor in [0.5, 2.0]
3. Random DC shift: add an offset sampled uniformly in [−10, +10] microvolts
4. Random time masking: set up to 25% of samples within the window to zero
5. Additive Gaussian noise: zero-mean with standard deviation < 0.2
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


def random_time_shift(window: np.ndarray, max_shift_sec: float = 2.5, 
                     fs: float = 200.0, p: float = 0.5) -> np.ndarray:
    """
    Apply random time shift to ECG window.
    
    Args:
        window: ECG window [n_samples, n_channels]
        max_shift_sec: Maximum shift in seconds (±)
        fs: Sampling frequency in Hz
        p: Probability of applying augmentation
        
    Returns:
        Time-shifted window [n_samples, n_channels]
    """
    if np.random.random() > p:
        return window
    
    n_samples = window.shape[0]
    max_shift_samples = int(max_shift_sec * fs)
    max_shift_samples = min(max_shift_samples, n_samples // 2)  # Don't shift more than half
    
    # Random shift in samples
    shift_samples = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    
    if shift_samples == 0:
        return window
    
    # Apply circular shift
    shifted_window = np.roll(window, shift_samples, axis=0)
    
    return shifted_window


def random_amplitude_scaling(window: np.ndarray, scale_range: Tuple[float, float] = (0.5, 2.0),
                           p: float = 0.5) -> np.ndarray:
    """
    Apply random amplitude scaling to ECG window.
    
    Args:
        window: ECG window [n_samples, n_channels]
        scale_range: Range of scaling factors (min, max)
        p: Probability of applying augmentation
        
    Returns:
        Amplitude-scaled window [n_samples, n_channels]
    """
    if np.random.random() > p:
        return window
    
    # Sample scaling factor
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    
    # Apply scaling
    scaled_window = window * scale_factor
    
    return scaled_window


def random_dc_shift(window: np.ndarray, shift_range_uv: Tuple[float, float] = (-10.0, 10.0),
                   p: float = 0.5) -> np.ndarray:
    """
    Apply random DC shift to ECG window.
    
    Args:
        window: ECG window [n_samples, n_channels]
        shift_range_uv: Range of DC shift in microvolts (min, max)
        p: Probability of applying augmentation
        
    Returns:
        DC-shifted window [n_samples, n_channels]
    """
    if np.random.random() > p:
        return window
    
    # Convert microvolts to millivolts (assuming window is in mV)
    shift_range_mv = (shift_range_uv[0] / 1000.0, shift_range_uv[1] / 1000.0)
    
    # Sample DC shift for each channel independently
    dc_shifts = np.random.uniform(shift_range_mv[0], shift_range_mv[1], 
                                 size=window.shape[1])
    
    # Apply DC shift
    shifted_window = window + dc_shifts[np.newaxis, :]
    
    return shifted_window


def random_time_masking(window: np.ndarray, max_mask_fraction: float = 0.25,
                       p: float = 0.5, contiguous: bool = True) -> np.ndarray:
    """
    Apply random time masking to ECG window.
    
    Args:
        window: ECG window [n_samples, n_channels]
        max_mask_fraction: Maximum fraction of samples to mask
        p: Probability of applying augmentation
        contiguous: Whether to use contiguous blocks (more realistic)
        
    Returns:
        Time-masked window [n_samples, n_channels]
    """
    if np.random.random() > p:
        return window
    
    n_samples = window.shape[0]
    max_mask_samples = int(n_samples * max_mask_fraction)
    
    if max_mask_samples == 0:
        return window
    
    # Sample number of samples to mask
    n_mask_samples = np.random.randint(1, max_mask_samples + 1)
    
    masked_window = window.copy()
    
    if contiguous:
        # Create contiguous masked region
        start_idx = np.random.randint(0, n_samples - n_mask_samples + 1)
        end_idx = start_idx + n_mask_samples
        masked_window[start_idx:end_idx, :] = 0
    else:
        # Create scattered masked samples
        mask_indices = np.random.choice(n_samples, size=n_mask_samples, replace=False)
        masked_window[mask_indices, :] = 0
    
    return masked_window


def add_gaussian_noise(window: np.ndarray, max_sigma: float = 0.2, 
                      p: float = 0.5) -> np.ndarray:
    """
    Add Gaussian noise to ECG window.
    
    Args:
        window: ECG window [n_samples, n_channels]
        max_sigma: Maximum standard deviation of noise
        p: Probability of applying augmentation
        
    Returns:
        Noisy window [n_samples, n_channels]
    """
    if np.random.random() > p:
        return window
    
    # Sample noise standard deviation
    sigma = np.random.uniform(0, max_sigma)
    
    # Generate noise
    noise = np.random.normal(0, sigma, size=window.shape)
    
    # Add noise
    noisy_window = window + noise
    
    return noisy_window


class ECGAugmentation:
    """
    ECG augmentation pipeline following Tang et al. 2022 specifications.
    """
    
    def __init__(self, 
                 time_shift_prob: float = 0.5,
                 amplitude_scale_prob: float = 0.5,
                 dc_shift_prob: float = 0.5,
                 time_mask_prob: float = 0.5,
                 noise_prob: float = 0.5,
                 max_shift_sec: float = 2.5,
                 scale_range: Tuple[float, float] = (0.5, 2.0),
                 dc_shift_range_uv: Tuple[float, float] = (-10.0, 10.0),
                 max_mask_fraction: float = 0.25,
                 max_noise_sigma: float = 0.2,
                 fs: float = 200.0):
        """
        Initialize ECG augmentation pipeline.
        
        Args:
            time_shift_prob: Probability of applying time shift
            amplitude_scale_prob: Probability of applying amplitude scaling
            dc_shift_prob: Probability of applying DC shift
            time_mask_prob: Probability of applying time masking
            noise_prob: Probability of applying noise
            max_shift_sec: Maximum time shift in seconds
            scale_range: Range of amplitude scaling factors
            dc_shift_range_uv: Range of DC shift in microvolts
            max_mask_fraction: Maximum fraction of samples to mask
            max_noise_sigma: Maximum noise standard deviation
            fs: Sampling frequency in Hz
        """
        self.time_shift_prob = time_shift_prob
        self.amplitude_scale_prob = amplitude_scale_prob
        self.dc_shift_prob = dc_shift_prob
        self.time_mask_prob = time_mask_prob
        self.noise_prob = noise_prob
        
        self.max_shift_sec = max_shift_sec
        self.scale_range = scale_range
        self.dc_shift_range_uv = dc_shift_range_uv
        self.max_mask_fraction = max_mask_fraction
        self.max_noise_sigma = max_noise_sigma
        self.fs = fs
    
    def __call__(self, window: np.ndarray) -> np.ndarray:
        """
        Apply augmentation pipeline to ECG window.
        
        Args:
            window: ECG window [n_samples, n_channels]
            
        Returns:
            Augmented window [n_samples, n_channels]
        """
        augmented = window.copy()
        
        # Apply augmentations in sequence
        # Order chosen to minimize interaction effects
        
        # 1. Time shift (affects temporal structure)
        augmented = random_time_shift(
            augmented, self.max_shift_sec, self.fs, self.time_shift_prob
        )
        
        # 2. Amplitude scaling (affects overall signal magnitude)
        augmented = random_amplitude_scaling(
            augmented, self.scale_range, self.amplitude_scale_prob
        )
        
        # 3. DC shift (affects baseline)
        augmented = random_dc_shift(
            augmented, self.dc_shift_range_uv, self.dc_shift_prob
        )
        
        # 4. Time masking (creates missing data)
        augmented = random_time_masking(
            augmented, self.max_mask_fraction, self.time_mask_prob
        )
        
        # 5. Gaussian noise (adds random variation)
        augmented = add_gaussian_noise(
            augmented, self.max_noise_sigma, self.noise_prob
        )
        
        return augmented
    
    def validate_augmented_signal(self, original: np.ndarray, 
                                 augmented: np.ndarray) -> bool:
        """
        Validate that augmented signal is physiologically reasonable.
        
        Args:
            original: Original ECG window
            augmented: Augmented ECG window
            
        Returns:
            True if augmented signal passes validation
        """
        # Check for extreme values (basic sanity check)
        max_reasonable_amplitude = 50.0  # mV (very conservative upper bound)
        
        if np.abs(augmented).max() > max_reasonable_amplitude:
            warnings.warn(f"Augmented signal has extreme amplitude: {np.abs(augmented).max():.2f} mV")
            return False
        
        # Check for NaN or infinite values
        if not np.isfinite(augmented).all():
            warnings.warn("Augmented signal contains NaN or infinite values")
            return False
        
        return True


def create_augmentation_pipeline(training: bool = True) -> Optional[ECGAugmentation]:
    """
    Create augmentation pipeline for training or validation/test.
    
    Args:
        training: If True, return augmentation pipeline. If False, return None.
        
    Returns:
        ECGAugmentation instance for training, None for validation/test
    """
    if not training:
        return None
    
    # Use Tang et al. 2022 specifications
    return ECGAugmentation(
        time_shift_prob=0.5,
        amplitude_scale_prob=0.5,
        dc_shift_prob=0.5,
        time_mask_prob=0.5,
        noise_prob=0.5,
        max_shift_sec=2.5,
        scale_range=(0.5, 2.0),
        dc_shift_range_uv=(-10.0, 10.0),
        max_mask_fraction=0.25,
        max_noise_sigma=0.2,
        fs=200.0
    )


if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing ECG augmentation pipeline...")
    
    # Create synthetic ECG window
    n_samples, n_channels = 1000, 12
    t = np.linspace(0, 5, n_samples)  # 5 seconds at 200 Hz
    
    # Generate synthetic ECG-like signals
    synthetic_ecg = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        # Simple synthetic ECG: combination of sine waves
        synthetic_ecg[:, ch] = (
            0.1 * np.sin(2 * np.pi * 1.2 * t) +  # Heart rate ~72 bpm
            0.05 * np.sin(2 * np.pi * 2.4 * t) +  # Harmonics
            0.02 * np.random.randn(n_samples)      # Noise
        )
    
    print(f"Original ECG shape: {synthetic_ecg.shape}")
    print(f"Original ECG range: [{synthetic_ecg.min():.3f}, {synthetic_ecg.max():.3f}]")
    
    # Create augmentation pipeline
    augmenter = create_augmentation_pipeline(training=True)
    
    # Apply augmentations multiple times
    n_tests = 5
    for i in range(n_tests):
        augmented = augmenter(synthetic_ecg)
        valid = augmenter.validate_augmented_signal(synthetic_ecg, augmented)
        print(f"Test {i+1}: Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}], "
              f"Valid: {valid}")
    
    print("Augmentation testing completed!")
