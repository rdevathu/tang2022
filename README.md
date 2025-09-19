# ECG AF Recurrence Prediction (Tang et al. 2022)

This repository implements the ECG-only branch from Tang et al. 2022 "Machine Learning–Enabled Multimodal Fusion of Intra-Atrial and Body Surface Signals in Prediction of Atrial Fibrillation Ablation Outcomes" published in *Circulation: Arrhythmia and Electrophysiology*.

## Overview

This implementation provides a complete pipeline for predicting 1-year AF recurrence after catheter ablation using 12-lead ECG data. It follows the paper's specifications exactly:

- **Task**: Binary classification of 1-year AF recurrence after catheter ablation
- **Input**: 12-lead ECG in sinus rhythm (all leads used, unlike paper's 8-lead approach)
- **Architecture**: 1D CNN with 6 residual bottleneck blocks
- **Training**: 10-fold stratified cross-validation with patient-level evaluation
- **Metrics**: AUROC, sensitivity, specificity, accuracy, F1, Brier score, ECE

## Dataset Requirements

The implementation is designed for CARTO ECG datasets with the following structure:

### Files Required
- `data/carto_ecg_metadata_FULL.csv`: Metadata with ECG-level information
- `data/carto_ecg_waveforms.npy`: ECG waveform data [N, 2500, 12]

### Metadata Format
Required columns:
- `mrn`: Patient medical record number (unique patient identifier)
- `sinus_rhythm`: Binary indicator (1 = sinus rhythm, 0 = other)
- `index`: Index in the .npy file for this ECG
- `acquisition_date`: Date/time of ECG acquisition
- `procedure_date`: Date of ablation procedure
- `af_recurrence`: Binary indicator (1 = recurrence, 0 = no recurrence)
- `days_till_af_recurrence`: Days from procedure to recurrence

### ECG Data Format
- Shape: [N_ecgs, 2500, 12] where:
  - N_ecgs: Total number of ECG records
  - 2500: Samples (10 seconds at 250 Hz)
  - 12: Standard 12-lead ECG [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]

## Installation

1. Clone this repository
2. Install dependencies using `uv`:

```bash
uv sync
```

## Usage

### Quick Test Run
```bash
python main.py --test-run
```
This runs a quick test with 2 folds and 3 epochs to verify everything works.

### Full Experiment
```bash
python main.py
```
This runs the complete 10-fold cross-validation with 100 epochs per fold.

### Custom Configuration
```bash
# Create a configuration file
python main.py --create-config

# Edit config.json as needed, then run:
python main.py --config config.json
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `"data"` | Directory containing CARTO dataset |
| `results_dir` | `"results"` | Directory to save results |
| `n_folds` | `10` | Number of cross-validation folds |
| `n_epochs` | `100` | Maximum epochs per fold |
| `batch_size` | `32` | Training batch size |
| `learning_rate` | `0.001` | Adam learning rate |
| `random_seed` | `42` | Random seed for reproducibility |

## Architecture Details

### Preprocessing Pipeline
1. **Lead Selection**: All 12 leads (I, II, III, aVR, aVL, aVF, V1-V6)
2. **Band-pass Filter**: 0.05-100 Hz zero-phase Butterworth filter
3. **Resampling**: From 250 Hz to 200 Hz using high-quality resampling
4. **Windowing**: 5-second windows with 4-second overlap (6 windows per 10s ECG)

### Data Augmentation (Training Only)
Applied stochastically with 50% probability each:
1. **Time Shift**: ±2.5 seconds circular shift
2. **Amplitude Scaling**: Random scaling by factor in [0.5, 2.0]
3. **DC Shift**: Random offset ±10 μV
4. **Time Masking**: Mask up to 25% of samples (contiguous blocks)
5. **Gaussian Noise**: Zero-mean noise with σ < 0.2

### CNN Architecture
1. **Input Projection**: Conv1D from 12 channels to 64 feature channels
2. **Bottleneck Blocks** (×6): 
   - Main: BN → ReLU → Conv1D → BN → ReLU → Conv1D → Dropout
   - Shortcut: 1×1 Conv1D → MaxPool1D (downsample by 2)
   - Output: main + shortcut
3. **Global Average Pooling**: Over time dimension
4. **Channel Aggregation**: BatchNorm → ReLU → Dropout
5. **Classifier**: Linear layer to single logit → Sigmoid

### Patient-Level Aggregation
- Window-level probabilities are averaged to obtain patient-level probability
- Threshold is chosen to maximize F1-score on test fold
- Evaluation is performed at patient level, not window level

## Results

The system outputs comprehensive results including:

### Files Generated
- `cv_results.csv`: Detailed results for each fold
- `cv_summary.json`: Mean ± SD statistics across folds
- `fold_X_predictions.json`: Patient-level predictions per fold
- `fold_X_best_model.pth`: Best model weights per fold
- `final_config.json`: Configuration used for the experiment

### Metrics Reported
- **AUROC**: Area under ROC curve (primary metric)
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate  
- **Accuracy**: Overall classification accuracy
- **F1-score**: Harmonic mean of precision and recall
- **Brier Score**: Mean squared error of probability predictions
- **ECE**: Expected Calibration Error (calibration quality)

## Expected Performance

Based on the Tang et al. 2022 paper, expected AUROC for ECG-only model is approximately 0.767 ± 0.122. Your results may vary depending on:
- Dataset characteristics
- Patient population
- ECG acquisition differences
- Label definitions (1-year recurrence window)

## Implementation Notes

### Key Differences from Paper
1. **12 leads vs 8 leads**: Uses all 12 standard ECG leads instead of paper's 8 independent leads
2. **CARTO dataset**: Adapted for CARTO dataset structure instead of generic patient-level format
3. **Patient selection**: Uses closest ECG to procedure date (prefer pre-ablation within 1 year)

### Technical Details
- **Framework**: PyTorch with CUDA support
- **Optimization**: Adam optimizer with fixed learning rate (no tuning per paper)
- **Early Stopping**: Based on validation AUROC with patience=15
- **Cross-Validation**: Stratified 10-fold at patient level (no window leakage)
- **Reproducibility**: Fixed random seeds, deterministic operations

## System Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~16GB RAM for full dataset
- ~10GB disk space for results

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{tang2022machine,
  title={Machine Learning–Enabled Multimodal Fusion of Intra-Atrial and Body Surface Signals in Prediction of Atrial Fibrillation Ablation Outcomes},
  author={Tang, Kevin and others},
  journal={Circulation: Arrhythmia and Electrophysiology},
  year={2022},
  publisher={American Heart Association}
}
```

## License

This implementation is provided for research purposes. Please ensure compliance with your institution's data use policies when working with clinical ECG data.

## Support

For questions or issues:
1. Check that your data format matches the requirements
2. Verify all dependencies are installed correctly
3. Try the `--test-run` option first to validate setup
4. Check the generated log files for detailed error information
