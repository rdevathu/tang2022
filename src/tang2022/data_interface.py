"""
Data interface for ECG dataset following Tang et al. 2022 specifications.

This module defines the expected data format and provides utilities for loading
and validating ECG data for AF recurrence prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod


class ECGLoader(ABC):
    """Abstract base class for ECG data loaders."""
    
    @abstractmethod
    def load_ecg(self, file_path: str) -> Tuple[np.ndarray, List[str], float]:
        """
        Load ECG data from file.
        
        Args:
            file_path: Path to ECG file
            
        Returns:
            Tuple of:
            - ECG data as numpy array [n_samples, n_leads]
            - List of lead names
            - Sampling rate in Hz
        """
        pass


class CSVECGLoader(ECGLoader):
    """Loader for ECG data stored in CSV format."""
    
    def __init__(self, lead_columns: List[str], sampling_rate: float):
        """
        Initialize CSV loader.
        
        Args:
            lead_columns: List of column names for ECG leads
            sampling_rate: Sampling rate in Hz
        """
        self.lead_columns = lead_columns
        self.sampling_rate = sampling_rate
    
    def load_ecg(self, file_path: str) -> Tuple[np.ndarray, List[str], float]:
        """Load ECG from CSV file."""
        df = pd.read_csv(file_path)
        ecg_data = df[self.lead_columns].values
        return ecg_data, self.lead_columns, self.sampling_rate


class WFDBECGLoader(ECGLoader):
    """Loader for WFDB format ECG data."""
    
    def load_ecg(self, file_path: str) -> Tuple[np.ndarray, List[str], float]:
        """Load ECG from WFDB format."""
        try:
            import wfdb
        except ImportError:
            raise ImportError("wfdb package required for WFDB format. Install with: uv add wfdb")
        
        record = wfdb.rdrecord(file_path.replace('.dat', '').replace('.hea', ''))
        return record.p_signal, record.sig_name, record.fs


def map_leads_to_standard(ecg_data: np.ndarray, lead_names: List[str]) -> np.ndarray:
    """
    Map ECG leads to standard order: [I, II, V1, V2, V3, V4, V5, V6].
    
    Args:
        ecg_data: ECG data [n_samples, n_leads]
        lead_names: List of lead names corresponding to columns in ecg_data
        
    Returns:
        ECG data with leads in standard order [n_samples, 8]
        
    Raises:
        ValueError: If required leads are missing
    """
    required_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Normalize lead names (handle common variations)
    lead_name_map = {
        'Lead I': 'I', 'LEAD_I': 'I', 'lead_I': 'I',
        'Lead II': 'II', 'LEAD_II': 'II', 'lead_II': 'II',
        'Lead V1': 'V1', 'LEAD_V1': 'V1', 'lead_V1': 'V1',
        'Lead V2': 'V2', 'LEAD_V2': 'V2', 'lead_V2': 'V2',
        'Lead V3': 'V3', 'LEAD_V3': 'V3', 'lead_V3': 'V3',
        'Lead V4': 'V4', 'LEAD_V4': 'V4', 'lead_V4': 'V4',
        'Lead V5': 'V5', 'LEAD_V5': 'V5', 'lead_V5': 'V5',
        'Lead V6': 'V6', 'LEAD_V6': 'V6', 'lead_V6': 'V6',
    }
    
    # Normalize lead names
    normalized_leads = []
    for name in lead_names:
        normalized_name = lead_name_map.get(name, name)
        normalized_leads.append(normalized_name)
    
    # Find indices for required leads
    lead_indices = []
    missing_leads = []
    
    for required_lead in required_leads:
        try:
            idx = normalized_leads.index(required_lead)
            lead_indices.append(idx)
        except ValueError:
            missing_leads.append(required_lead)
    
    if missing_leads:
        raise ValueError(f"Missing required leads: {missing_leads}. "
                        f"Available leads: {normalized_leads}")
    
    # Extract and reorder leads
    return ecg_data[:, lead_indices]


class ECGDataset:
    """
    Dataset class for ECG data following Tang et al. 2022 specifications.
    """
    
    def __init__(self, metadata_path: str, ecg_loader: ECGLoader, 
                 data_root: Optional[str] = None):
        """
        Initialize ECG dataset.
        
        Args:
            metadata_path: Path to metadata CSV file
            ecg_loader: ECG loader instance
            data_root: Root directory for ECG files (if paths in metadata are relative)
        """
        self.metadata = pd.read_csv(metadata_path)
        self.ecg_loader = ecg_loader
        self.data_root = Path(data_root) if data_root else None
        
        # Validate metadata format
        self._validate_metadata()
    
    def _validate_metadata(self):
        """Validate metadata format."""
        required_columns = ['patient_id', 'label', 'ecg_path', 'sampling_rate_hz']
        missing_columns = [col for col in required_columns if col not in self.metadata.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in metadata: {missing_columns}")
        
        # Validate labels are 0/1
        unique_labels = self.metadata['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError(f"Labels must be 0/1, found: {unique_labels}")
        
        print(f"Dataset loaded: {len(self.metadata)} patients")
        print(f"Label distribution: {self.metadata['label'].value_counts().to_dict()}")
    
    def get_patient_data(self, patient_id: str) -> Tuple[np.ndarray, int, float]:
        """
        Get ECG data for a specific patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Tuple of:
            - ECG data [n_samples, 8] (leads: I, II, V1-V6)
            - Label (0/1)
            - Original sampling rate
        """
        patient_row = self.metadata[self.metadata['patient_id'] == patient_id]
        if len(patient_row) == 0:
            raise ValueError(f"Patient {patient_id} not found")
        
        patient_row = patient_row.iloc[0]
        
        # Construct full path
        ecg_path = patient_row['ecg_path']
        if self.data_root and not Path(ecg_path).is_absolute():
            ecg_path = self.data_root / ecg_path
        
        # Load ECG data
        ecg_data, lead_names, fs = self.ecg_loader.load_ecg(str(ecg_path))
        
        # Map to standard leads
        ecg_standard = map_leads_to_standard(ecg_data, lead_names)
        
        return ecg_standard, int(patient_row['label']), fs
    
    def get_patient_ids(self) -> List[str]:
        """Get list of all patient IDs."""
        return self.metadata['patient_id'].tolist()
    
    def get_labels(self) -> np.ndarray:
        """Get all labels for stratified splitting."""
        return self.metadata['label'].values


def create_example_metadata(output_path: str, n_patients: int = 100):
    """
    Create an example metadata file for reference.
    
    Args:
        output_path: Path to save example metadata CSV
        n_patients: Number of example patients to create
    """
    np.random.seed(42)
    
    # Generate example data
    patient_ids = [f"patient_{i:04d}" for i in range(n_patients)]
    labels = np.random.choice([0, 1], size=n_patients, p=[0.7, 0.3])  # 30% recurrence rate
    ecg_paths = [f"ecg_data/{pid}.csv" for pid in patient_ids]
    sampling_rates = np.random.choice([250, 500, 1000], size=n_patients)
    
    metadata = pd.DataFrame({
        'patient_id': patient_ids,
        'label': labels,
        'ecg_path': ecg_paths,
        'sampling_rate_hz': sampling_rates,
        'datetime_ecg': pd.date_range('2020-01-01', periods=n_patients, freq='D')
    })
    
    metadata.to_csv(output_path, index=False)
    print(f"Example metadata saved to {output_path}")
    return metadata


if __name__ == "__main__":
    # Create example metadata for reference
    create_example_metadata("data/example_metadata.csv")
