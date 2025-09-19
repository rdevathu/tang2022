## Goal

Produce an ECG-only model that matches the “ECG branch” in Tang et al. 2022 (Circulation: Arrhythmia and Electrophysiology) as closely as possible. This includes:
-  Data requirements and preprocessing
-  Windowing/augmentation
-  CNN architecture and training procedure
-  Cross-validation and evaluation
-  Patient-level inference and thresholding
-  Reproducibility details

Below is a precise, step-by-step implementation outline and LLM-executable instructions to plug in any ECG dataset with the same assumptions as the paper.

---

## Assumptions to Mirror the Paper Exactly

-  Task: Binary classification of 1-year AF recurrence after catheter ablation.
-  Input modality: 12-lead ECG, but use 8 independent channels only (I, II, V1–V6).
-  Rhythm: Sinus rhythm ECG closest to (within 1 year before) the ablation. If unavailable, a post-ablation sinus rhythm ECG may be used, but for your dataset you should prefer pre-ablation if available.
-  Sampling: Resample to 200 Hz.
-  Band-pass: 0.05–100 Hz.
-  Windowing: 5-second windows with 4-second overlap.
-  Architecture: 1D CNN with residual bottleneck blocks operating first over time, then a 1D conv over channels, identical to Supplemental Methods with 6 bottleneck blocks for ECG.
-  Training: Adam optimizer; no hyperparameter tuning; data augmentation as specified; 10-fold stratified cross-validation at patient level; optimize AUROC.
-  Aggregation: Average window-level probabilities to patient-level probability.
-  Metrics: AUROC, sensitivity, specificity, accuracy, F1; probability threshold chosen to maximize F1 on test folds; also report Brier score and ECE if possible.

---

## Data Interface Specification (what your dataset must provide)

For each patient:
-  Unique patient_id
-  Outcome label: recurrence_1y ∈ {0,1}
-  One sinus rhythm 12-lead ECG recording close to ablation (prefer ≤ 1 year pre-ablation)
    - Sampling rate known (will be resampled to 200 Hz)
    - 12 leads available; we will extract leads: I, II, V1, V2, V3, V4, V5, V6
    - Units: millivolts preferred. If different, provide conversion to mV.

Recommended file organization:
-  A metadata CSV with columns:
    - patient_id, label (0/1), ecg_path, sampling_rate_hz, datetime_ecg
-  ECG files per patient: array shape [n_samples, n_leads] with lead order documented. If the file is in a vendor format, provide a loader to produce a numpy array and map lead names.

Lead mapping function must output an ordered array: [I, II, V1, V2, V3, V4, V5, V6].

---

## Preprocessing Pipeline (ECG-only, exactly as in the paper)

Implement the following steps for each patient’s ECG:

1) Lead selection
-  Extract 8 independent channels: I, II, V1–V6
-  Discard other leads (III, aVR, aVL, aVF), because linear dependencies can be learned by the model from I and II.

2) Band-pass filter
-  Apply a zero-phase band-pass filter from 0.05 Hz to 100 Hz per channel.

3) Resample
-  Resample all channels to 200 Hz using high-quality resampling (e.g., scipy.signal.resample_poly).

4) Windowing
-  Partition the multi-lead signal into 5-second windows with 4-second overlap.
    - At 200 Hz, 5-second window = 1,000 samples.
    - Overlap 4 seconds = 800-sample overlap; stride = 200 samples.
-  For each window, build an array of shape [time=1000, channels=8].

5) Optional padding (as in Supplemental Fig I)
-  If needed by your pooling schedule, you may pad time dimension to 1024 with trailing zeros for uniform downsampling by max-pooling. Keep a consistent approach across all windows.

6) Normalization
-  The paper does not explicitly state per-window normalization for ECG, beyond filtering and resampling. To match the paper, do not apply z-scoring. If your dataset scale differs wildly (e.g., microvolts), convert to mV consistently before training.

---

## Data Augmentation (apply on-the-fly during training)

Use exactly the five augmentations described, adapted to ECG windows. Apply stochastically per window (with reasonable per-op probability like 0.5 unless conflicts occur). Ensure augmentations do not create physiologically invalid signals.

-  Random time shift within the 5-sec window by up to ±2.5 sec (circular or pad with zeros; keep output length at 5 sec).
-  Random amplitude scaling: multiply signal by a factor in [0.5, 2.0].
-  Random DC shift in microvolts: add an offset sampled uniformly in [−10, +10] microvolts (i.e., ±0.01 mV).
-  Random time masking: set up to 25% of samples within the window to zero (contiguous blocks or scattered; prefer contiguous blocks for realism).
-  Additive Gaussian noise: zero-mean with standard deviation < 0.2 (units must match your signal; in mV this is large; if your post-filter signal RMS is small, consider a proportional sigma, but to mirror the paper, cap at 0.2 in the same units).

Note: The paper states these augmentations “did not result in invalid signals.” Keep the final amplitude within a physiologic range where possible.

---

## CNN Architecture (ECG-only branch)

Match Supplemental Methods I exactly but with 6 bottleneck blocks for ECG. The structure:

-  Input: window tensor [L, C] where L=1000 (or 1024 padded), C=8
-  Reformat to [batch, channels_in, time] as needed by your DL framework. Keep channel dimension representing the feature channels along time for time-convs.

Time-dimension residual bottleneck blocks (×6):
-  Each bottleneck block has two branches:
    - Main branch:
        - BatchNorm
        - ReLU
        - 1D convolution along time (kernel size per Attia et al. 2019; if unspecified, use 7 or 5 and keep consistent)
        - BatchNorm
        - ReLU
        - 1D convolution along time
    - Shortcut branch:
        - 1×1 convolution along time to match channels if needed
        - Max-pooling (downsample time by 2)
-  Outputs from both branches are added.
-  Insert dropout between bottleneck blocks.
-  After 6 blocks, apply average pooling over time.

Channel-dimension convolution:
-  1D convolution over the channel dimension (i.e., across the 8 ECG leads/features) to aggregate spatial information, followed by BatchNorm and ReLU.

Classifier head:
-  Fully connected layer to a single logit
-  Sigmoid activation for probability

Important notes:
-  Use the same kernel sizes, output channels, dropout probabilities as Attia et al. 2019 where referenced by Tang et al. When exact values are missing, follow a standard residual-1D stack akin to Attia:
    - Example typical config (if you need concrete defaults consistent with Attia-like ECG models):
        - Initial conv channels: 32–64, doubling every 1–2 blocks
        - Kernel size 5–7
        - Dropout p=0.2–0.5
        - Final feature dimension: 128 before FC
    - However, to remain faithful to Tang et al., do not perform ad hoc hyperparameter tuning; fix your chosen values and keep them constant.

Activation and normalization:
-  Use ReLU and BatchNorm as specified.
-  Use dropout between blocks and before FC.

Padding:
-  If you padded the time dimension to 1024 pre-network, ensure max-pool strides halve time length at each bottleneck so that shapes remain integers.

---

## Training Protocol

-  Framework: PyTorch (as in paper); single GPU is fine.
-  Optimizer: Adam
-  Loss: Binary cross-entropy with logits (or BCE if you apply sigmoid earlier)
-  Learning rate: As in paper, they didn’t tune; use a standard default like 1e-3 and cosine decay or step LR only if necessary. Since they report “did not tune hyperparameters,” prefer a fixed LR 1e-3.
-  Batch size: Choose based on GPU memory (e.g., 32 windows).
-  Epochs: Train until convergence or a fixed number (e.g., 50–100 epochs); early stopping on validation AUROC is acceptable within each fold.
-  Objective: Optimize AUROC. In practice, minimize BCE loss and select best epoch by highest validation AUROC.

Data splitting:
-  Stratified 10-fold cross-validation at patient level.
    - Randomly assign patients to 10 folds with equal recurrence prevalence where possible.
    - For fold i:
        - Train on folds != i
        - Test on fold i
-  Ensure no window leakage across folds for the same patient.

On-the-fly augmentations:
-  Apply only to training windows, not validation/test.

Class imbalance:
-  If imbalanced, consider class weighting in BCE or balanced sampling of windows. The paper does not report special handling; to mirror, avoid extensive rebalancing; stratified folds help.

---

## Inference and Patient-Level Aggregation

-  For each patient in the test fold:
    - Run the ECG through preprocessing and windowing.
    - Obtain window-level probabilities from the CNN.
    - Average all window probabilities to yield the patient-level probability of recurrence.
        - This matches Supplemental Methods I: “probabilities from all 5-sec windows of the same patient were averaged.”

-  Threshold selection:
    - Compute metrics on the test fold using a probability threshold chosen by “highest F1-score on the 10 fold test sets.”
    - To mirror the paper strictly, for each fold, after generating probabilities for that test fold, choose the threshold that maximizes F1 for that fold’s predictions. Report metrics at that threshold.

---

## Evaluation Metrics and Calibration

For each fold, compute:
-  AUROC (primary)
-  Sensitivity, specificity, accuracy, F1-score at the F1-optimal threshold
-  Optionally calibration:
    - Brier score: mean squared error between predicted probability and label
    - Expected Calibration Error (ECE):
        - Bin predictions into equal-width probability bins (e.g., 10–20 bins)
        - Compute weighted average of |accuracy − confidence| across bins

Aggregate reporting:
-  Report mean ± SD across 10 folds for all metrics.

---

## Reproducibility

-  Fix random seeds (e.g., Python, NumPy, PyTorch) per fold.
-  Log fold assignments, train/val/test splits, best epoch, and chosen thresholds.
-  Save model weights per fold.

---

## LLM-Executable Step-by-Step Instructions

You can give these directly to an LLM agent with Python execution capability.

1) Load metadata
-  Read CSV with columns: patient_id, label, ecg_path, sampling_rate_hz.
-  Filter to patients with an available sinus rhythm ECG.
-  Ensure labels are 0/1 for 1-year AF recurrence.

2) Build splits
-  Perform patient-level stratified 10-fold split using label.

3) Define signal I/O
-  Implement load_ecg(file) → np.ndarray [n_samples, n_leads], lead_names: list[str].
-  Implement map_leads_to_I_II_V1_V6(array, lead_names) → np.ndarray [n_samples, 8].
    - Leads in order: I, II, V1, V2, V3, V4, V5, V6.
    - Validate presence; if missing, skip patient or map from vector relations if accurate (not recommended; better to require leads).

4) Preprocessing functions
-  bandpass_filter_0p05_100(signal, fs_in) → filtered signal
-  resample_to_200hz(signal, fs_in) → signal_200hz
-  window_signal(signal_200hz, window_sec=5, overlap_sec=4, fs=200) → list of [1000, 8] windows
-  optional_pad_to_1024(window) if your pooling requires it.
-  Note: Do not z-score or per-window normalize, to match paper.

5) Data augmentation transforms (train only)
-  random_time_shift(window, max_shift_sec=2.5)
-  random_scale(window, scale_range=[0.5, 2.0])
-  random_dc_shift_microvolts(window, range_uv=[-10, 10])  // if your units are mV, add ±0.01 mV
-  random_time_mask(window, max_frac=0.25)
-  add_gaussian_noise(window, sigma<0.2)
-  Compose into an on-the-fly augmentation pipeline with random application per op.

6) Dataset and Dataloader
-  Training dataset yields augmented windows and labels (label repeated per window).
-  Validation/test datasets yield non-augmented windows.
-  Maintain mapping from patient to its windows for aggregation.

7) Model definition
-  Implement the ECG CNN with:
    - 6 residual bottleneck blocks over time (each: BN → ReLU → Conv1D → BN → ReLU → Conv1D, plus shortcut 1×1 conv + max-pool; add outputs).
    - Dropout between blocks and before FC.
    - Average pooling over time after the last time-block stack.
    - Conv1D over channels (across the 8 leads), BN, ReLU.
    - FC to 1 logit; sigmoid at inference time.

8) Training loop per fold
-  Initialize model, Adam(lr=1e-3), BCEWithLogitsLoss.
-  Train for N epochs (e.g., 50–100), evaluating on held-out test fold after each epoch to track AUROC.
-  Save best epoch by validation AUROC on a held-out portion of training data (e.g., split training set into train/val at patient level) or simply by test AUROC if you strictly follow the paper’s description (they do 10-fold CV; typical practice is to select by val AUROC within the training folds to avoid peeking at test).
-  After training, run inference on test fold; store window-level probabilities and aggregate to patient-level by mean.

9) Thresholding and metrics
-  Using test fold patient-level probabilities and labels:
    - Sweep thresholds from 0 to 1 (e.g., 0.00 to 1.00 in 0.001 steps).
    - Compute F1-score at each threshold; select threshold that maximizes F1.
    - At the chosen threshold, compute sensitivity, specificity, accuracy, F1.
    - Compute AUROC independent of threshold.
    - Optionally compute Brier score and ECE.

10) Aggregate results
-  Repeat for all 10 folds.
-  Compute mean ± SD for AUROC, sensitivity, specificity, accuracy, F1, and calibration measures.

11) Save artifacts
-  Per fold: model weights, fold indices, patient-level predictions, chosen threshold, metrics.
-  Global: summary CSV/table of fold metrics and their mean ± SD.

---

## Minimal PyTorch Skeleton (pseudo-code)

Note: Indentation uses 4 spaces exactly as required.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout_p=0.3):
        super().__init__()
        pad = kernel_size // 2
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.Dropout(p=dropout_p),
        )
        self.short = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, C, T]
        y = self.main(x)
        y = self.pool(y)
        s = self.short(x)
        return y + s

class ECGNet(nn.Module):
    def __init__(self, in_channels=8, base_ch=64, blocks=6, dropout_p=0.3):
        super().__init__()
        chs = [base_ch]*(blocks+1)
        self.proj = nn.Conv1d(in_channels, chs[0], kernel_size=7, padding=3)
        blks = []
        for i in range(blocks):
            blks.append(BottleneckBlock(chs[i], chs[i+1], kernel_size=7, dropout_p=dropout_p))
        self.time_stack = nn.Sequential(*blks)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # pool over time to length 1

        # Channel-dimension conv: treat channels as features to aggregate across leads.
        # After time pooling, tensor is [B, C_feat, 1]. We can reinterpret "channel conv"
        # as a linear layer across C_feat. Alternatively, if you implement channel-conv explicitly,
        # you must permute dims to apply a conv over the channel axis. Here we approximate with Linear.
        self.channel_agg = nn.Sequential(
            nn.BatchNorm1d(chs[-1]),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(chs[-1], 1)
        )

    def forward(self, x):
        # x: [B, T, C] -> convert to [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.time_stack(x)
        x = self.avg_pool(x).squeeze(-1)  # [B, C_feat]
        x = self.channel_agg(x)
        logit = self.fc(x).squeeze(-1)
        return logit
```

Notes:
-  The above approximates the “channel-dimension convolution” after time pooling with a linear projection across features (equivalent to a 1×1 conv across feature dimension). If you prefer a strict channel-conv layer, reshape accordingly and apply Conv1d over the “channel” axis after a suitable permutation.
-  Keep blocks=6 for the ECG model, per the paper.

---

## Plugging In Your Dataset: Checklist

-  Do you have an ECG per patient with known sampling rate?
-  Can you map to leads I, II, V1–V6?
-  Can you resample to 200 Hz and filter 0.05–100 Hz?
-  Can you generate 5-second windows with 4-second overlap?
-  Are labels patient-level (1-year AF recurrence yes/no)?
-  Can you perform patient-level stratified 10-fold CV?

If yes to all, you can run the pipeline above unchanged.

---

## Expected Performance

The paper reports for ECG-only:
-  AUROC ≈ 0.767 ± 0.122 across 10 folds.

Your results will vary depending on cohort, ECG acquisition differences, and label definitions. To remain faithful, avoid hyperparameter tuning and maintain the described augmentations and architecture.

---

## Reporting Template

Use this table format to report results (mean ± SD across folds):

|:---|:---|:---|:---|:---|
| AUROC | Sensitivity | Specificity | Accuracy | F1-score |

Optionally include calibration:

|:---|:---|
| Brier score | ECE |

---

## Notes on Deviations

-  Do not introduce additional features, clinical variables, or imaging.
-  Do not include other ECG leads beyond I, II, V1–V6.
-  Do not apply feature engineering; the CNN learns representations directly from raw filtered signals.
-  Keep hyperparameters fixed; the paper explicitly avoided tuning for CNNs.

---

## Troubleshooting

-  If overfitting:
    - Verify augmentations are active for training only.
    - Increase dropout modestly (e.g., from 0.3 to 0.4) but keep architecture fixed.
    - Ensure no patient leakage across folds.
-  If window counts vary widely per patient:
    - That is expected; the model averages probabilities at inference.
-  If lead order is incorrect:
    - AUROC will degrade. Double-check lead mapping.

---

## References Embedded in the Design

-  Tang et al. 2022: Machine Learning–Enabled Multimodal Fusion of Intra-Atrial and Body Surface Signals in Prediction of Atrial Fibrillation Ablation Outcomes. Circulation: Arrhythmia and Electrophysiology.
-  Supplemental Methods and Figures from the same paper detailing CNN design, augmentations, windowing, and training.
-  Attia et al. 2019: As architectural reference for the bottleneck residual ECG CNN with 1D convs over time and a channel aggregation stage.

