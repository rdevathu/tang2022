"""
PyTorch dataset classes for ECG data following Tang et al. 2022 specifications.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional, Tuple, Dict
from .carto_data_interface import CARTODataset
from .preprocessing import preprocess_patient_ecg
from .augmentation import ECGAugmentation


class ECGWindowDataset(Dataset):
    """
    PyTorch dataset for ECG windows with patient-level labels.
    
    Each sample is a preprocessed ECG window with the patient's outcome label.
    Maintains mapping from windows back to patients for aggregation.
    """
    
    def __init__(self, 
                 carto_dataset: CARTODataset,
                 patient_ids: List[str],
                 augmentation: Optional[ECGAugmentation] = None,
                 target_fs: float = 200.0,
                 window_sec: float = 5.0,
                 overlap_sec: float = 4.0,
                 pad_to_1024: bool = False):
        """
        Initialize ECG window dataset.
        
        Args:
            carto_dataset: CARTO dataset instance
            patient_ids: List of patient MRNs to include
            augmentation: Augmentation pipeline (None for val/test)
            target_fs: Target sampling frequency
            window_sec: Window duration in seconds
            overlap_sec: Overlap duration in seconds
            pad_to_1024: Whether to pad windows to 1024 samples
        """
        self.carto_dataset = carto_dataset
        self.patient_ids = patient_ids
        self.augmentation = augmentation
        self.target_fs = target_fs
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.pad_to_1024 = pad_to_1024
        
        # Preprocess all patients and create window index
        self._create_window_index()
    
    def _create_window_index(self):
        """Create index mapping from window index to (patient_id, window_idx, label)."""
        self.window_index = []
        self.patient_windows = {}  # patient_id -> list of windows
        
        print(f"Preprocessing {len(self.patient_ids)} patients...")
        
        for patient_id in self.patient_ids:
            # Get patient ECG and label
            ecg_data, label = self.carto_dataset.get_patient_ecg(patient_id)
            fs_original = self.carto_dataset.get_sampling_rate()
            
            # Preprocess ECG into windows
            windows = preprocess_patient_ecg(
                ecg_data, fs_original, self.target_fs,
                window_sec=self.window_sec, overlap_sec=self.overlap_sec,
                pad_to_1024=self.pad_to_1024
            )
            
            # Store windows for this patient
            self.patient_windows[patient_id] = windows
            
            # Add to global window index
            for window_idx in range(len(windows)):
                self.window_index.append((patient_id, window_idx, label))
        
        print(f"Created {len(self.window_index)} windows from {len(self.patient_ids)} patients")
        
        # Print label distribution at window level
        labels = [item[2] for item in self.window_index]
        label_counts = {0: labels.count(0), 1: labels.count(1)}
        print(f"Window-level label distribution: {label_counts}")
    
    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.window_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single window sample.
        
        Args:
            idx: Window index
            
        Returns:
            Tuple of:
            - ECG window tensor [time, channels]
            - Label tensor [1]
            - Patient ID string
        """
        patient_id, window_idx, label = self.window_index[idx]
        
        # Get window data
        window = self.patient_windows[patient_id][window_idx]  # [time, channels]
        
        # Apply augmentation if training
        if self.augmentation is not None:
            window = self.augmentation(window)
        
        # Convert to tensors
        window_tensor = torch.tensor(window, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return window_tensor, label_tensor, patient_id
    
    def get_patient_windows(self, patient_id: str) -> List[int]:
        """Get all window indices for a specific patient."""
        return [i for i, (pid, _, _) in enumerate(self.window_index) if pid == patient_id]


def create_dataloaders(carto_dataset: CARTODataset,
                      train_patients: List[str],
                      test_patients: List[str],
                      batch_size: int = 32,
                      num_workers: int = 4,
                      augmentation: Optional[ECGAugmentation] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        carto_dataset: CARTO dataset instance
        train_patients: List of training patient IDs
        test_patients: List of test patient IDs
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        augmentation: Augmentation pipeline for training (None for test)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = ECGWindowDataset(
        carto_dataset, train_patients, augmentation=augmentation
    )
    
    test_dataset = ECGWindowDataset(
        carto_dataset, test_patients, augmentation=None  # No augmentation for test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def aggregate_patient_predictions(predictions: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Aggregate window-level predictions to patient-level by averaging.
    
    Args:
        predictions: Dict mapping patient_id -> list of window probabilities
        
    Returns:
        Dict mapping patient_id -> average probability
    """
    patient_probs = {}
    
    for patient_id, window_probs in predictions.items():
        if len(window_probs) > 0:
            patient_probs[patient_id] = np.mean(window_probs)
        else:
            patient_probs[patient_id] = 0.0  # Default for patients with no windows
    
    return patient_probs


if __name__ == "__main__":
    # Test dataset creation
    from .carto_data_interface import load_carto_dataset
    from .augmentation import create_augmentation_pipeline
    
    print("Testing ECG dataset creation...")
    
    # Load CARTO dataset
    carto_dataset = load_carto_dataset()
    
    # Get patient IDs
    all_patients = carto_dataset.get_patient_ids()
    print(f"Total patients: {len(all_patients)}")
    
    # Split into train/test (simple split for testing)
    n_train = int(0.8 * len(all_patients))
    train_patients = all_patients[:n_train]
    test_patients = all_patients[n_train:]
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Create augmentation
    augmentation = create_augmentation_pipeline(training=True)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        carto_dataset, train_patients, test_patients,
        batch_size=8, num_workers=0, augmentation=augmentation
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Test a batch
    for batch_idx, (windows, labels, patient_ids) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Windows shape: {windows.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Patient IDs: {patient_ids[:3]}...")  # First 3 IDs
        print(f"  Label distribution: {labels.unique(return_counts=True)}")
        break
    
    print("Dataset testing completed!")
