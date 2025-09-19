# Final Model Training for ECG AF Recurrence Prediction

This document describes the updated training methodology that replaces cross-validation with a proper train-test split approach to build the optimal final model for deployment.

## Overview

The original implementation used 10-fold cross-validation for model evaluation. While this approach is excellent for assessing model performance and generalization, it doesn't produce a single final model optimized for deployment. 

This updated approach implements a **train-validation-test split methodology** to build the best possible final model that can learn ECG patterns for AF recurrence prediction.

## Key Improvements

### 1. Proper Data Splitting Strategy
- **Training Set (60%)**: Used to train the final model
- **Validation Set (20%)**: Used for early stopping and hyperparameter optimization
- **Test Set (20%)**: Held-out for final unbiased evaluation

### 2. Single Optimized Model
- Trains one final model instead of 10 separate fold models
- Uses all available training data efficiently
- Optimized for best possible performance on deployment

### 3. Robust Early Stopping
- Monitors validation AUROC for early stopping
- Increased patience (20 epochs) for better convergence
- Prevents overfitting while maximizing learning

### 4. Stratified Splits
- Maintains class balance across all splits
- Ensures representative samples in each partition
- Preserves patient-level integrity (no patient appears in multiple splits)

## Usage

### Quick Start - Train Final Model

```bash
# Train final model with default settings
python train_final_model.py

# Train with custom configuration
python train_final_model.py --config config_final_model.json

# Quick test run
python train_final_model.py --test-run
```

### Custom Parameters

```bash
# Custom data splits and training parameters
python train_final_model.py \
    --test-size 0.15 \
    --val-size 0.15 \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Using Main Script with Final Model Option

```bash
# Use main.py with final model training
python main.py --final-model --test-size 0.2 --val-size 0.2
```

## Configuration

### Recommended Configuration (`config_final_model.json`)

```json
{
  "data_dir": "data",
  "results_dir": "final_model_results",
  "test_size": 0.2,
  "val_size": 0.2,
  "n_epochs": 150,
  "batch_size": 32,
  "learning_rate": 0.001,
  "patience": 20,
  "random_seed": 42,
  "training_mode": "final_model"
}
```

### Key Parameters

- **test_size**: Fraction for test set (0.15-0.25 recommended)
- **val_size**: Fraction for validation set (0.15-0.25 recommended) 
- **n_epochs**: Maximum training epochs (100-200 for convergence)
- **patience**: Early stopping patience (15-25 epochs)
- **learning_rate**: Adam learning rate (1e-4 to 1e-3)

## Output Files

After training, the following files are created in `results_dir`:

### Model Files
- `final_model.pth`: PyTorch model checkpoint (ready for deployment)
- `final_model_results.json`: Complete results and metrics
- `final_training_config.json`: Configuration used for training

### Analysis Files
- `final_test_predictions.json`: Test set predictions and patient IDs
- `training_history.json`: Training progress (losses, metrics)

### Analysis and Visualization

```bash
# Analyze final model results
python analyze_final_model.py --results-dir final_model_results

# Generate analysis with saved plots
python analyze_final_model.py --results-dir final_model_results --save-plots

# Summary only
python analyze_final_model.py --summary-only
```

## Performance Expectations

### Expected Improvements Over Cross-Validation
1. **Single Deployable Model**: Ready for production use
2. **Better Convergence**: More epochs on full dataset
3. **Optimized Performance**: Early stopping on validation set
4. **Deployment Ready**: No model selection required

### Typical Performance Ranges
- **AUROC**: 0.75-0.85 (dataset dependent)
- **Sensitivity**: 0.70-0.85
- **Specificity**: 0.70-0.85
- **F1 Score**: 0.65-0.80

## Training Strategy Comparison

| Aspect | Cross-Validation | Final Model Training |
|--------|------------------|---------------------|
| Purpose | Model evaluation | Model deployment |
| Output | 10 separate models | 1 optimized model |
| Data usage | 90% train, 10% test per fold | 60% train, 20% val, 20% test |
| Training time | 10x model training | 1x model training |
| Model selection | Average performance | Best single model |
| Deployment | Requires ensemble or selection | Direct deployment |

## Best Practices

### 1. Data Quality
- Ensure clean, preprocessed ECG data
- Verify patient-level labels are correct
- Check for data leakage between splits

### 2. Training Parameters
- Start with recommended configuration
- Monitor validation metrics closely
- Use early stopping to prevent overfitting

### 3. Evaluation
- Always evaluate on held-out test set
- Check calibration of predictions
- Analyze failure cases for insights

### 4. Model Deployment
- Use `final_model.pth` for inference
- Implement same preprocessing pipeline
- Monitor performance on new data

## Troubleshooting

### Common Issues

1. **Poor Convergence**
   - Increase `n_epochs` (try 200-300)
   - Decrease `learning_rate` (try 5e-4)
   - Increase `patience` (try 25-30)

2. **Overfitting**
   - Reduce model complexity
   - Increase validation size
   - Add regularization

3. **Low Performance**
   - Check data quality and labels
   - Verify preprocessing consistency
   - Consider data augmentation

### Memory Issues
```bash
# Reduce batch size for limited memory
python train_final_model.py --batch-size 16

# Use CPU if GPU memory insufficient
CUDA_VISIBLE_DEVICES="" python train_final_model.py
```

## Integration with Existing Pipeline

The final model training approach is fully compatible with the existing codebase:

- Uses same data preprocessing pipeline
- Compatible with existing model architecture
- Maintains same evaluation metrics
- Preserves patient-level aggregation

## Migration from Cross-Validation

To switch from cross-validation to final model training:

1. Use `train_final_model.py` instead of `main.py`
2. Update configuration for final model parameters
3. Evaluate using `analyze_final_model.py`
4. Deploy using `final_model.pth`

This approach provides a more practical and deployment-ready solution while maintaining the scientific rigor of the original Tang et al. 2022 methodology.