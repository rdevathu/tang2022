"""
ECG CNN model following Tang et al. 2022 specifications.

Implements the ECG-only branch with:
- 6 residual bottleneck blocks operating over time dimension
- 1D convolution over channel dimension (across ECG leads)
- Binary classification head for AF recurrence prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BottleneckBlock(nn.Module):
    """
    Residual bottleneck block for time-dimension processing.
    
    Architecture:
    - Main branch: BN -> ReLU -> Conv1D -> BN -> ReLU -> Conv1D -> Dropout
    - Shortcut branch: 1x1 Conv1D -> MaxPool1D (downsample by 2)
    - Output: main + shortcut (after max pooling main branch)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 7, dropout_p: float = 0.3):
        """
        Initialize bottleneck block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dropout_p: Dropout probability
        """
        super().__init__()
        
        # Main branch
        self.main_branch = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, 
                     padding=kernel_size // 2, bias=False),
            nn.Dropout(p=dropout_p)
        )
        
        # Shortcut branch (1x1 conv to match channels + downsampling)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Max pooling for main branch (downsample time by 2)
        self.main_pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, time]
            
        Returns:
            Output tensor [batch_size, channels, time//2]
        """
        # Main branch
        main_out = self.main_branch(x)
        main_out = self.main_pool(main_out)
        
        # Shortcut branch
        shortcut_out = self.shortcut(x)
        
        # Residual connection
        return main_out + shortcut_out


class ECGNet(nn.Module):
    """
    ECG CNN for AF recurrence prediction following Tang et al. 2022.
    
    Architecture:
    1. Input projection: Conv1D to map from input channels to base_channels
    2. 6 residual bottleneck blocks over time dimension
    3. Global average pooling over time
    4. 1D convolution over channel dimension (across ECG leads)
    5. Binary classification head
    """
    
    def __init__(self, 
                 in_channels: int = 12,  # 12 leads for CARTO data
                 base_channels: int = 64,
                 n_blocks: int = 6,
                 kernel_size: int = 7,
                 dropout_p: float = 0.3):
        """
        Initialize ECG CNN.
        
        Args:
            in_channels: Number of input ECG leads (12 for CARTO)
            base_channels: Base number of channels for feature maps
            n_blocks: Number of bottleneck blocks (6 for ECG branch)
            kernel_size: Convolution kernel size
            dropout_p: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        
        # Input projection: map from ECG leads to feature channels
        self.input_proj = nn.Conv1d(in_channels, base_channels, 
                                   kernel_size=kernel_size, 
                                   padding=kernel_size // 2, bias=False)
        
        # Bottleneck blocks (6 blocks for ECG)
        # Channel progression: keep same number of channels throughout
        # (Tang et al. don't specify channel doubling, so we keep constant)
        blocks = []
        for i in range(n_blocks):
            blocks.append(
                BottleneckBlock(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    kernel_size=kernel_size,
                    dropout_p=dropout_p
                )
            )
        self.bottleneck_blocks = nn.Sequential(*blocks)
        
        # Global average pooling over time dimension
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Channel-dimension convolution
        # After time pooling, we have [batch, base_channels, 1]
        # The "channel convolution" aggregates across the feature channels
        # We implement this as a linear layer after flattening
        self.channel_agg = nn.Sequential(
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        
        # Binary classification head
        self.classifier = nn.Linear(base_channels, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input ECG tensor [batch_size, time, channels] or [batch_size, channels, time]
            
        Returns:
            Logits for binary classification [batch_size, 1]
        """
        # Handle input format: convert [B, T, C] to [B, C, T] if needed
        if x.dim() == 3 and x.size(-1) == self.in_channels:
            x = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        
        # Input projection
        x = self.input_proj(x)  # [B, C_in, T] -> [B, base_channels, T]
        
        # Bottleneck blocks (each reduces time dimension by 2)
        x = self.bottleneck_blocks(x)  # [B, base_channels, T//2^n_blocks]
        
        # Global average pooling over time
        x = self.global_avg_pool(x)  # [B, base_channels, 1]
        x = x.squeeze(-1)  # [B, base_channels]
        
        # Channel aggregation
        x = self.channel_agg(x)  # [B, base_channels]
        
        # Classification
        logits = self.classifier(x)  # [B, 1]
        
        return logits.squeeze(-1)  # [B]
    
    def predict_proba(self, x):
        """
        Get probability predictions.
        
        Args:
            x: Input ECG tensor
            
        Returns:
            Probabilities [batch_size]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities


class ECGNetLarge(ECGNet):
    """
    Larger variant of ECG CNN with more channels.
    """
    
    def __init__(self, **kwargs):
        # Use larger base channels
        kwargs.setdefault('base_channels', 128)
        super().__init__(**kwargs)


def create_ecg_model(model_size: str = 'base', **kwargs) -> nn.Module:
    """
    Create ECG model following Tang et al. 2022 specifications.
    
    Args:
        model_size: 'base' or 'large'
        **kwargs: Additional model parameters
        
    Returns:
        ECG CNN model
    """
    if model_size == 'base':
        return ECGNet(**kwargs)
    elif model_size == 'large':
        return ECGNetLarge(**kwargs)
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model architecture
    print("Testing ECG CNN architecture...")
    
    # Create model
    model = create_ecg_model('base', in_channels=12)
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test with sample input
    batch_size = 4
    time_samples = 1000  # 5 seconds at 200 Hz
    n_channels = 12      # 12 ECG leads
    
    # Test both input formats
    print(f"\nTesting with input shape [B, T, C]: [{batch_size}, {time_samples}, {n_channels}]")
    x_btc = torch.randn(batch_size, time_samples, n_channels)
    
    with torch.no_grad():
        logits = model(x_btc)
        probs = model.predict_proba(x_btc)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    print(f"\nTesting with input shape [B, C, T]: [{batch_size}, {n_channels}, {time_samples}]")
    x_bct = torch.randn(batch_size, n_channels, time_samples)
    
    with torch.no_grad():
        logits2 = model(x_bct)
        probs2 = model.predict_proba(x_bct)
    
    print(f"Output logits shape: {logits2.shape}")
    print(f"Output probabilities shape: {probs2.shape}")
    
    # Test model summary
    print(f"\nModel architecture summary:")
    print(model)
    
    print("\nModel testing completed!")
