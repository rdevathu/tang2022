"""
Data interface for CARTO ECG dataset following Tang et al. 2022 specifications.

This module handles the complex CARTO dataset structure where:
- ECGs are stored in a single .npy file (16616, 2500, 12)
- Metadata is ECG-level, not patient-level
- Patients are identified by MRN
- Labels need to be derived from af_recurrence and days_till_af_recurrence
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import warnings


class CARTODataset:
    """
    Dataset class for CARTO ECG data following Tang et al. 2022 specifications.
    """
    
    def __init__(self, metadata_path: str, ecg_data_path: str, 
                 max_days_to_recurrence: int = 365):
        """
        Initialize CARTO ECG dataset.
        
        Args:
            metadata_path: Path to metadata CSV file
            ecg_data_path: Path to ECG numpy file
            max_days_to_recurrence: Maximum days to consider as positive outcome (default: 365)
        """
        self.metadata = pd.read_csv(metadata_path)
        self.ecg_data = np.load(ecg_data_path)
        self.max_days_to_recurrence = max_days_to_recurrence
        
        print(f"Loaded {len(self.metadata)} ECG records from {ecg_data_path}")
        print(f"ECG data shape: {self.ecg_data.shape}")
        
        # Validate data consistency
        self._validate_data()
        
        # Process patient-level data
        self.patient_data = self._create_patient_level_data()
        
        print(f"Processed {len(self.patient_data)} unique patients")
    
    def _validate_data(self):
        """Validate data consistency."""
        assert len(self.metadata) == self.ecg_data.shape[0], \
            f"Metadata rows ({len(self.metadata)}) != ECG data rows ({self.ecg_data.shape[0]})"
        
        assert self.ecg_data.shape[1:] == (2500, 12), \
            f"Expected ECG shape (2500, 12), got {self.ecg_data.shape[1:]}"
        
        print("Data validation passed")
    
    def _create_patient_level_data(self) -> pd.DataFrame:
        """
        Create patient-level dataset from ECG-level metadata.
        
        For each patient (MRN):
        1. Find all sinus rhythm ECGs
        2. Identify ECGs closest to procedure date (prefer pre-ablation within 1 year)
        3. Determine outcome label based on AF recurrence within max_days_to_recurrence
        
        Returns:
            DataFrame with columns: mrn, ecg_index, label, acquisition_date, procedure_date
        """
        # Filter to sinus rhythm ECGs only
        sinus_ecgs = self.metadata[self.metadata['sinus_rhythm'] == 1].copy()
        print(f"Sinus rhythm ECGs: {len(sinus_ecgs)}")
        
        # Filter to ECGs with procedure dates and outcome data
        with_outcomes = sinus_ecgs[
            sinus_ecgs['procedure_date'].notna() & 
            sinus_ecgs['af_recurrence'].notna()
        ].copy()
        print(f"Sinus ECGs with procedure dates and outcomes: {len(with_outcomes)}")
        
        # Convert date columns to datetime
        with_outcomes['acquisition_date'] = pd.to_datetime(with_outcomes['acquisition_date'])
        with_outcomes['procedure_date'] = pd.to_datetime(with_outcomes['procedure_date'])
        
        # Calculate days between ECG and procedure (negative = pre-procedure)
        with_outcomes['days_to_procedure'] = (
            with_outcomes['procedure_date'] - with_outcomes['acquisition_date']
        ).dt.days
        
        patient_records = []
        
        # Group by patient (MRN)
        for mrn, patient_ecgs in with_outcomes.groupby('mrn'):
            # Determine outcome label
            af_recurrence = patient_ecgs['af_recurrence'].iloc[0]  # Should be same for all ECGs of a patient
            days_till_recurrence = patient_ecgs['days_till_af_recurrence'].iloc[0]
            
            # Label logic: 1 if AF recurrence within max_days_to_recurrence, 0 otherwise
            if af_recurrence == 1.0 and pd.notna(days_till_recurrence):
                label = 1 if days_till_recurrence <= self.max_days_to_recurrence else 0
            else:
                label = 0  # No recurrence or recurrence beyond time window
            
            # Find best ECG for this patient
            # Prefer pre-procedure ECGs within 1 year, then closest to procedure
            pre_procedure = patient_ecgs[patient_ecgs['days_to_procedure'] >= 0]
            within_year = pre_procedure[pre_procedure['days_to_procedure'] <= 365]
            
            if len(within_year) > 0:
                # Use closest pre-procedure ECG within 1 year
                best_ecg = within_year.loc[within_year['days_to_procedure'].idxmin()]
            elif len(pre_procedure) > 0:
                # Use closest pre-procedure ECG (even if > 1 year)
                best_ecg = pre_procedure.loc[pre_procedure['days_to_procedure'].idxmin()]
            else:
                # Use closest post-procedure ECG
                best_ecg = patient_ecgs.loc[patient_ecgs['days_to_procedure'].abs().idxmin()]
            
            patient_records.append({
                'mrn': mrn,
                'ecg_index': best_ecg['index'],
                'label': label,
                'acquisition_date': best_ecg['acquisition_date'],
                'procedure_date': best_ecg['procedure_date'],
                'days_to_procedure': best_ecg['days_to_procedure'],
                'af_recurrence': af_recurrence,
                'days_till_af_recurrence': days_till_recurrence
            })
        
        patient_df = pd.DataFrame(patient_records)
        
        # Print summary statistics
        print(f"\nPatient-level summary:")
        print(f"Total patients: {len(patient_df)}")
        print(f"Label distribution: {patient_df['label'].value_counts().to_dict()}")
        print(f"Days to procedure - Mean: {patient_df['days_to_procedure'].mean():.1f}, "
              f"Median: {patient_df['days_to_procedure'].median():.1f}")
        
        return patient_df
    
    def get_patient_ecg(self, mrn) -> Tuple[np.ndarray, int]:
        """
        Get ECG data and label for a specific patient.
        
        Args:
            mrn: Patient MRN (can be string or numeric)
            
        Returns:
            Tuple of:
            - ECG data [2500, 12] (all 12 leads)
            - Label (0/1)
        """
        # Convert mrn to numeric if it's a string
        if isinstance(mrn, str):
            try:
                mrn = int(mrn)
            except ValueError:
                pass  # Keep as string if conversion fails
        
        patient_row = self.patient_data[self.patient_data['mrn'] == mrn]
        if len(patient_row) == 0:
            raise ValueError(f"Patient {mrn} not found")
        
        patient_row = patient_row.iloc[0]
        ecg_index = int(patient_row['ecg_index'])
        label = int(patient_row['label'])
        
        ecg_data = self.ecg_data[ecg_index]  # Shape: [2500, 12]
        
        return ecg_data, label
    
    def get_patient_ids(self) -> List[str]:
        """Get list of all patient MRNs."""
        return [str(mrn) for mrn in self.patient_data['mrn'].tolist()]
    
    def get_labels(self) -> np.ndarray:
        """Get all labels for stratified splitting."""
        return self.patient_data['label'].values
    
    def get_sampling_rate(self) -> float:
        """Get sampling rate (250 Hz for all CARTO ECGs)."""
        return 250.0
    
    def get_lead_names(self) -> List[str]:
        """Get lead names for 12-lead ECG."""
        # Standard 12-lead order
        return ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def create_tang_compatible_metadata(self, output_path: str):
        """
        Create Tang et al. 2022 compatible metadata CSV.
        
        Args:
            output_path: Path to save compatible metadata
        """
        compatible_data = []
        
        for _, row in self.patient_data.iterrows():
            compatible_data.append({
                'patient_id': str(row['mrn']),
                'label': row['label'],
                'ecg_path': f"carto_index_{row['ecg_index']}.npy",  # Virtual path
                'sampling_rate_hz': 250,
                'datetime_ecg': row['acquisition_date'],
                'procedure_date': row['procedure_date'],
                'days_to_procedure': row['days_to_procedure']
            })
        
        compatible_df = pd.DataFrame(compatible_data)
        compatible_df.to_csv(output_path, index=False)
        print(f"Tang-compatible metadata saved to {output_path}")
        
        return compatible_df
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        stats = {
            'total_ecg_records': len(self.metadata),
            'sinus_rhythm_ecgs': len(self.metadata[self.metadata['sinus_rhythm'] == 1]),
            'unique_patients': len(self.patient_data),
            'label_distribution': self.patient_data['label'].value_counts().to_dict(),
            'recurrence_rate': self.patient_data['label'].mean(),
            'ecg_shape': self.ecg_data.shape,
            'sampling_rate': 250,
            'duration_seconds': 10,
            'leads': self.get_lead_names(),
            'days_to_procedure_stats': {
                'mean': self.patient_data['days_to_procedure'].mean(),
                'median': self.patient_data['days_to_procedure'].median(),
                'min': self.patient_data['days_to_procedure'].min(),
                'max': self.patient_data['days_to_procedure'].max(),
            }
        }
        
        return stats


def load_carto_dataset(data_dir: str = "data") -> CARTODataset:
    """
    Load CARTO dataset from data directory.
    
    Args:
        data_dir: Directory containing carto_ecg_metadata_FULL.csv and carto_ecg_waveforms.npy
        
    Returns:
        CARTODataset instance
    """
    metadata_path = Path(data_dir) / "carto_ecg_metadata_FULL.csv"
    ecg_path = Path(data_dir) / "carto_ecg_waveforms.npy"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG data file not found: {ecg_path}")
    
    return CARTODataset(str(metadata_path), str(ecg_path))


if __name__ == "__main__":
    # Load and analyze CARTO dataset
    dataset = load_carto_dataset()
    
    # Print statistics
    stats = dataset.get_dataset_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create Tang-compatible metadata
    dataset.create_tang_compatible_metadata("data/tang_compatible_metadata.csv")
    
    # Test loading a patient
    patient_ids = dataset.get_patient_ids()
    if patient_ids:
        test_mrn = patient_ids[0]
        ecg, label = dataset.get_patient_ecg(test_mrn)
        print(f"\nTest patient {test_mrn}:")
        print(f"  ECG shape: {ecg.shape}")
        print(f"  Label: {label}")
        print(f"  ECG range: [{ecg.min():.2f}, {ecg.max():.2f}]")
