"""
Tang et al. 2022 ECG AF Recurrence Prediction Implementation.

This package implements the ECG-only branch from Tang et al. 2022 
"Machine Learning–Enabled Multimodal Fusion of Intra-Atrial and Body Surface 
Signals in Prediction of Atrial Fibrillation Ablation Outcomes"

Key components:
- CARTO data interface for ECG loading and patient-level processing
- Preprocessing pipeline (filtering, resampling, windowing)
- Data augmentation following paper specifications
- CNN architecture with residual bottleneck blocks
- Training framework with 10-fold cross-validation
- Comprehensive evaluation metrics
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "ECG AF Recurrence Prediction (Tang et al. 2022)"

# Import key classes and functions
from .carto_data_interface import CARTODataset, load_carto_dataset
from .model import ECGNet, create_ecg_model
from .training import ECGTrainer, run_cross_validation
from .preprocessing import preprocess_patient_ecg
from .augmentation import ECGAugmentation, create_augmentation_pipeline
from .dataset import ECGWindowDataset, create_dataloaders

__all__ = [
    "CARTODataset",
    "load_carto_dataset", 
    "ECGNet",
    "create_ecg_model",
    "ECGTrainer",
    "run_cross_validation",
    "preprocess_patient_ecg",
    "ECGAugmentation",
    "create_augmentation_pipeline",
    "ECGWindowDataset",
    "create_dataloaders"
]
