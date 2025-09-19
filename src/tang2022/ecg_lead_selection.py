"""
ECG lead selection following Tang et al. 2022 specifications.

The paper specifies using 8 independent ECG channels: I, II, V1-V6
(excluding III, aVR, aVL, aVF due to linear dependencies).
"""

import numpy as np
from typing import List, Tuple


def select_8_independent_leads(ecg_data: np.ndarray, lead_names: List[str]) -> np.ndarray:
    """
    Select 8 independent ECG leads following Tang et al. 2022.
    
    Args:
        ecg_data: ECG data [n_samples, n_leads]
        lead_names: List of lead names corresponding to columns
        
    Returns:
        ECG data with 8 independent leads [n_samples, 8] in order: I, II, V1, V2, V3, V4, V5, V6
        
    Raises:
        ValueError: If required leads are missing
    """
    # Standard 12-lead order for CARTO data
    standard_12_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # 8 independent leads as specified in Tang et al. 2022
    required_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Normalize lead names (handle common variations)
    lead_name_map = {
        'Lead I': 'I', 'LEAD_I': 'I', 'lead_I': 'I',
        'Lead II': 'II', 'LEAD_II': 'II', 'lead_II': 'II',
        'Lead III': 'III', 'LEAD_III': 'III', 'lead_III': 'III',
        'Lead aVR': 'aVR', 'LEAD_aVR': 'aVR', 'lead_aVR': 'aVR',
        'Lead aVL': 'aVL', 'LEAD_aVL': 'aVL', 'lead_aVL': 'aVL',
        'Lead aVF': 'aVF', 'LEAD_aVF': 'aVF', 'lead_aVF': 'aVF',
        'Lead V1': 'V1', 'LEAD_V1': 'V1', 'lead_V1': 'V1',
        'Lead V2': 'V2', 'LEAD_V2': 'V2', 'lead_V2': 'V2',
        'Lead V3': 'V3', 'LEAD_V3': 'V3', 'lead_V3': 'V3',
        'Lead V4': 'V4', 'LEAD_V4': 'V4', 'lead_V4': 'V4',
        'Lead V5': 'V5', 'LEAD_V5': 'V5', 'lead_V5': 'V5',
        'Lead V6': 'V6', 'LEAD_V6': 'V6', 'lead_V6': 'V6',
    }
    
    # If lead_names is empty, assume standard 12-lead order
    if not lead_names or len(lead_names) != ecg_data.shape[1]:
        if ecg_data.shape[1] == 12:
            lead_names = standard_12_leads
        else:
            raise ValueError(f"Cannot infer lead names for ECG with {ecg_data.shape[1]} channels")
    
    # Normalize lead names
    normalized_leads = []
    for name in lead_names:
        normalized_name = lead_name_map.get(name, name)
        normalized_leads.append(normalized_name)
    
    # Find indices for the 8 required leads
    lead_indices = []
    missing_leads = []
    
    for required_lead in required_leads:
        try:
            idx = normalized_leads.index(required_lead)
            lead_indices.append(idx)
        except ValueError:
            missing_leads.append(required_lead)
    
    if missing_leads:
        raise ValueError(f"Missing required leads for Tang et al. 2022: {missing_leads}. "
                        f"Available leads: {normalized_leads}")
    
    # Extract the 8 independent leads in the correct order
    selected_ecg = ecg_data[:, lead_indices]
    
    print(f"Selected 8 independent ECG leads: {required_leads}")
    print(f"From original {len(normalized_leads)} leads: {normalized_leads}")
    
    return selected_ecg


def get_tang_2022_lead_names() -> List[str]:
    """Get the 8 lead names used in Tang et al. 2022."""
    return ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


if __name__ == "__main__":
    # Test lead selection
    print("Testing ECG lead selection for Tang et al. 2022...")
    
    # Simulate 12-lead ECG data
    n_samples = 2500  # 10 seconds at 250 Hz
    n_leads = 12
    dummy_ecg = np.random.randn(n_samples, n_leads)
    
    # Standard 12-lead names
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    print(f"Original ECG shape: {dummy_ecg.shape}")
    print(f"Original leads: {lead_names}")
    
    # Select 8 independent leads
    selected_ecg = select_8_independent_leads(dummy_ecg, lead_names)
    
    print(f"Selected ECG shape: {selected_ecg.shape}")
    print(f"Expected shape: ({n_samples}, 8)")
    
    assert selected_ecg.shape == (n_samples, 8), f"Wrong shape: {selected_ecg.shape}"
    print("✅ Lead selection test passed!")
