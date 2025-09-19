"""
ECG validation utilities to detect and filter out problematic ECG signals.

This module provides functions to:
1. Detect ECGs with zero/flat leads
2. Detect other signal quality issues
3. Validate ECG signals before using them for training
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings


def is_lead_zero_or_flat(lead_signal: np.ndarray, 
                        zero_threshold: float = 1e-3,  # More realistic for ECG
                        flat_threshold: float = 1e-3,  # More realistic for ECG
                        min_variation_samples: int = 10) -> bool:
    """
    Check if a lead is essentially zero or flat (no meaningful variation).
    
    This function is designed to catch obviously problematic leads like:
    - Leads that are all zeros (disconnected)
    - Leads that are constant values (flat lines)
    - Leads with extremely low variation
    
    Args:
        lead_signal: Single lead ECG signal [n_samples]
        zero_threshold: Threshold for considering values as zero
        flat_threshold: Threshold for standard deviation to be considered flat
        min_variation_samples: Minimum number of samples that should vary
        
    Returns:
        True if lead is zero or flat, False otherwise
    """
    # First check: if most values are essentially zero (disconnected lead)
    zero_samples = np.abs(lead_signal) < zero_threshold
    zero_fraction = np.sum(zero_samples) / len(lead_signal)
    
    if zero_fraction > 0.95:  # More than 95% of samples are zero
        return True
    
    # Second check: if signal has extremely low variation (flat line)
    signal_std = np.std(lead_signal)
    signal_range = np.max(lead_signal) - np.min(lead_signal)
    
    # A real ECG should have some variation - even a very flat ECG should have some noise
    if signal_std < flat_threshold and signal_range < (flat_threshold * 10):
        return True
    
    # Third check: if signal is constant (exactly the same value)
    unique_values = len(np.unique(np.round(lead_signal, 6)))  # Round to avoid floating point issues
    if unique_values <= 1:
        return True
    
    return False


def validate_ecg_quality(ecg_data: np.ndarray, 
                        lead_names: List[str] = None,
                        zero_threshold: float = 1e-6,
                        flat_threshold: float = 1e-3) -> Tuple[bool, Dict]:
    """
    Validate ECG signal quality and detect problematic leads.
    
    Args:
        ecg_data: ECG data [n_samples, n_leads]
        lead_names: Names of ECG leads
        zero_threshold: Threshold for considering values as zero
        flat_threshold: Threshold for standard deviation to be considered flat
        
    Returns:
        Tuple of:
        - is_valid: True if ECG passes quality checks
        - quality_info: Dictionary with quality metrics and issues
    """
    n_samples, n_leads = ecg_data.shape
    
    if lead_names is None:
        lead_names = [f"Lead_{i}" for i in range(n_leads)]
    
    quality_info = {
        'n_samples': n_samples,
        'n_leads': n_leads,
        'problematic_leads': [],
        'zero_leads': [],
        'flat_leads': [],
        'signal_ranges': [],
        'signal_stds': [],
        'issues': []
    }
    
    is_valid = True
    
    # Check each lead
    for lead_idx in range(n_leads):
        lead_signal = ecg_data[:, lead_idx]
        lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"Lead_{lead_idx}"
        
        # Calculate basic statistics
        lead_min, lead_max = np.min(lead_signal), np.max(lead_signal)
        lead_range = lead_max - lead_min
        lead_std = np.std(lead_signal)
        
        quality_info['signal_ranges'].append(lead_range)
        quality_info['signal_stds'].append(lead_std)
        
        # Check if lead is zero or flat
        if is_lead_zero_or_flat(lead_signal, zero_threshold, flat_threshold):
            quality_info['problematic_leads'].append(lead_name)
            is_valid = False
            
            # Determine if it's zero or just flat
            zero_samples = np.abs(lead_signal) < zero_threshold
            zero_fraction = np.sum(zero_samples) / len(lead_signal)
            
            if zero_fraction > 0.95:
                quality_info['zero_leads'].append(lead_name)
                quality_info['issues'].append(f"{lead_name}: {zero_fraction:.1%} zero samples")
            else:
                quality_info['flat_leads'].append(lead_name)
                quality_info['issues'].append(f"{lead_name}: flat signal (std={lead_std:.3f})")
    
    # Additional quality checks
    
    # Check for extreme values that might indicate artifacts
    extreme_threshold = 10000  # Very large values might indicate artifacts
    if np.any(np.abs(ecg_data) > extreme_threshold):
        quality_info['issues'].append(f"Extreme values detected (>{extreme_threshold})")
        # Don't mark as invalid for extreme values alone, just note it
    
    # Check for NaN or infinite values
    if not np.isfinite(ecg_data).all():
        quality_info['issues'].append("NaN or infinite values detected")
        is_valid = False
    
    return is_valid, quality_info


def find_best_valid_ecg(patient_ecgs: List[Tuple], 
                       ecg_data_loader,
                       lead_names: List[str] = None) -> Tuple[int, Dict]:
    """
    Find the best valid ECG from a list of ECGs for a patient.
    
    Args:
        patient_ecgs: List of (ecg_index, days_to_procedure, acquisition_date) tuples
                     sorted by preference (closest to procedure first)
        ecg_data_loader: Function to load ECG data given an index
        lead_names: Names of ECG leads
        
    Returns:
        Tuple of:
        - best_ecg_index: Index of best valid ECG, or -1 if none found
        - validation_results: Dictionary with validation info for all tested ECGs
    """
    validation_results = {
        'tested_ecgs': [],
        'valid_ecgs': [],
        'invalid_ecgs': [],
        'selected_ecg': None
    }
    
    for ecg_index, days_to_procedure, acquisition_date in patient_ecgs:
        # Load ECG data
        ecg_data = ecg_data_loader(ecg_index)
        
        # Validate quality
        is_valid, quality_info = validate_ecg_quality(ecg_data, lead_names)
        
        test_result = {
            'ecg_index': ecg_index,
            'days_to_procedure': days_to_procedure,
            'acquisition_date': acquisition_date,
            'is_valid': is_valid,
            'quality_info': quality_info
        }
        
        validation_results['tested_ecgs'].append(test_result)
        
        if is_valid:
            validation_results['valid_ecgs'].append(test_result)
            # Return the first valid ECG (they're sorted by preference)
            validation_results['selected_ecg'] = test_result
            return ecg_index, validation_results
        else:
            validation_results['invalid_ecgs'].append(test_result)
    
    # No valid ECG found
    return -1, validation_results


if __name__ == "__main__":
    # Test ECG validation
    print("Testing ECG validation...")
    
    # Create test ECGs
    n_samples, n_leads = 2500, 8
    
    # Test 1: Normal ECG-like signal
    normal_ecg = np.zeros((n_samples, n_leads))
    t = np.linspace(0, 10, n_samples)
    for lead in range(n_leads):
        # Generate realistic ECG-like signal
        normal_ecg[:, lead] = (
            1.0 * np.sin(2 * np.pi * 1.2 * t) +      # Heart rate ~72 bpm
            0.5 * np.sin(2 * np.pi * 2.4 * t) +      # Harmonics
            0.1 * np.random.randn(n_samples)          # Noise
        )
    
    is_valid, info = validate_ecg_quality(normal_ecg)
    print(f"Normal ECG - Valid: {is_valid}, Issues: {info['issues']}")
    
    # Test 2: ECG with one zero lead
    zero_lead_ecg = normal_ecg.copy()
    zero_lead_ecg[:, 0] = 0  # Make first lead all zeros
    is_valid, info = validate_ecg_quality(zero_lead_ecg)
    print(f"Zero lead ECG - Valid: {is_valid}, Issues: {info['issues']}")
    
    # Test 3: ECG with one flat lead
    flat_lead_ecg = normal_ecg.copy()
    flat_lead_ecg[:, 1] = 1.5  # Make second lead constant
    is_valid, info = validate_ecg_quality(flat_lead_ecg)
    print(f"Flat lead ECG - Valid: {is_valid}, Issues: {info['issues']}")
    
    # Test 4: ECG with extreme values
    extreme_ecg = normal_ecg.copy()
    extreme_ecg[100:110, 2] = 15000  # Add some extreme values
    is_valid, info = validate_ecg_quality(extreme_ecg)
    print(f"Extreme values ECG - Valid: {is_valid}, Issues: {info['issues']}")
    
    print("ECG validation testing completed!")
