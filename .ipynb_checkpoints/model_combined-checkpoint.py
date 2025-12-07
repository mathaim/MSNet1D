"""
model.py
========
Neural network architectures for LOS prediction.

Contains:
- FocalLoss: Loss function for imbalanced classification
- MultiScaleBlock: Multi-path feature extraction block
- ASPPModule: Atrous Spatial Pyramid Pooling for 1D data
- MSNet1D: Complete multi-scale network with dual heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the loss contribution from easy examples,
    focusing training on hard examples.
    
    Args:
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultiScaleBlock(nn.Module):
    """
    Multi-scale feature extraction block.
    
    Uses 3 parallel paths with different depths to capture
    feature interactions at multiple scales:
    - Path 1: Direct transformation (1 layer)
    - Path 2: Medium depth (2 layers)
    - Path 3: Deep with bottleneck (3 layers)
    
    Includes skip connection for residual learning.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        
        # Path 1: Direct (1 layer)
        self.path1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Path 2: Medium depth (2 layers)
        self.path2 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Path 3: Deep with bottleneck (3 layers)
        bottleneck = in_features // 2
        self.path3 = nn.Sequential(
            nn.Linear(in_features, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Skip connection
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()
        
        # Combine paths
        self.combine = nn.Sequential(
            nn.Linear(out_features * 3, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Parallel paths
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        
        # Skip connection
        skip = self.skip(x)
        
        # Concatenate and combine
        combined = torch.cat([p1, p2, p3], dim=1)
        out = self.combine(combined)
        
        # Residual connection
        return out + skip


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling adapted for 1D tabular data.
    
    Original ASPP uses different dilation rates for multi-scale context.
    This 1D adaptation uses different network depths instead.
    
    4 parallel branches:
    - Branch 1: 1 layer (local features)
    - Branch 2: 2 layers (medium scale)
    - Branch 3: 3 layers (large scale)
    - Branch 4: 1 layer (global context)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
    
    Reference:
        Chen et al., "DeepLab: Semantic Image Segmentation", TPAMI 2017
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        hidden = in_features // 4
        
        # Branch 1: Local (1 layer)
        self.branch1 = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 2: Medium (2 layers)
        self.branch2 = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 3: Deep (3 layers)
        self.branch3 = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 4: Global (1 layer)
        self.branch4 = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Combine all branches
        self.combine = nn.Sequential(
            nn.Linear(hidden * 4, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        combined = torch.cat([b1, b2, b3, b4], dim=1)
        return self.combine(combined)


class MSNet1D(nn.Module):
    """
    Multi-Scale Network for 1D tabular data (Linear version).
    
    Architecture:
        Input → Projection → MSBlock1 → MSBlock2 → ASPP → Shared → Heads
    
    Features:
    - Multi-scale feature extraction via parallel paths
    - ASPP module for multi-level context
    - Dual-head output for multi-task learning
    - Skip connections for gradient flow
    
    Args:
        input_dim: Number of input features
        num_classes: Number of classification classes (default: 4)
        dropout: Dropout rate (default: 0.3)
    
    Returns:
        Tuple of (regression_output, classification_output)
        - regression_output: [batch_size, 1] - LOS prediction
        - classification_output: [batch_size, num_classes] - class logits
    """
    
    def __init__(self, input_dim, num_classes=4, dropout=0.1):
        super().__init__()
        
        # Input projection: variable input → fixed 128 dimensions
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale blocks
        self.ms_block1 = MultiScaleBlock(128, 64, dropout)
        self.ms_block2 = MultiScaleBlock(64, 64, dropout)
        
        # ASPP module
        self.aspp = ASPPModule(64, 64)
        
        # Shared representation layer
        self.shared = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Regression head (LOS in hours)
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Classification head (LOS quartile)
        self.cls_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.input_proj(x)
        x = self.ms_block1(x)
        x = self.ms_block2(x)
        x = self.aspp(x)
        
        # Shared representation
        embedding = self.shared(x)
        
        # Task-specific outputs
        reg_out = self.reg_head(embedding)
        cls_out = self.cls_head(embedding)
        
        return reg_out, cls_out
    
    def get_embedding(self, x):
        """Get the shared embedding representation."""
        x = self.input_proj(x)
        x = self.ms_block1(x)
        x = self.ms_block2(x)
        x = self.aspp(x)
        return self.shared(x)


# =============================================================================
# CONV1D VERSION WITH MAXPOOLING (Closer to original MSNet paper)
# =============================================================================

class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale feature extraction block using Conv1D + MaxPool.
    
    Treats each feature as a "time step" in a sequence, with channels
    representing different learned representations.
    
    Uses 3 parallel paths with different kernel sizes to capture
    patterns at multiple scales, plus MaxPooling for downsampling.
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        
        # Path 1: Small kernel (local patterns)
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Path 2: Medium kernel (medium-range patterns)
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Path 3: Large kernel (long-range patterns)
        self.path3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MaxPool for downsampling (key difference from linear version)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Combine paths (3 * out_channels -> out_channels)
        self.combine = nn.Sequential(
            nn.Conv1d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        ) if in_channels != out_channels else nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x):
        # Parallel paths
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        
        # Concatenate along channel dimension
        combined = torch.cat([p1, p2, p3], dim=1)
        
        # Combine and pool
        out = self.combine(combined)
        out = self.pool(out)
        
        # Skip connection (also pooled)
        skip = self.skip(x)
        
        # Handle dimension mismatch if needed
        if out.shape != skip.shape:
            # Adjust skip to match output size
            if out.shape[2] != skip.shape[2]:
                skip = skip[:, :, :out.shape[2]]
        
        return out + skip


class ASPPConvModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling using dilated Conv1D.
    
    Uses different dilation rates to capture multi-scale context,
    similar to the original ASPP for images.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        hidden = in_channels // 4
        
        # Branch 1: 1x1 conv (no dilation)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 2: 3x3 conv with dilation=2
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 3: 3x3 conv with dilation=4
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 4: Global average pooling
        self.branch4_pool = nn.AdaptiveAvgPool1d(1)
        self.branch4_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Combine all branches
        self.combine = nn.Sequential(
            nn.Conv1d(hidden * 4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        size = x.shape[2]
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Global pooling branch - pool, conv, then upsample back
        b4 = self.branch4_pool(x)
        b4 = self.branch4_conv(b4)
        b4 = nn.functional.interpolate(b4, size=size, mode='nearest')
        
        # Concatenate all branches
        combined = torch.cat([b1, b2, b3, b4], dim=1)
        
        return self.combine(combined)


class MSNet1D_Conv(nn.Module):
    """
    Multi-Scale Network for 1D tabular data using Conv1D + MaxPooling.
    
    This version is closer to the original MSNet paper architecture:
    - Uses Conv1d instead of Linear layers
    - Includes MaxPooling for hierarchical feature extraction
    - Uses dilated convolutions in ASPP module
    
    The input features are treated as a 1D "sequence" where each feature
    is a time step, and we learn channel representations.
    
    Architecture:
        Input → Expand to channels → MSConvBlock1 → MSConvBlock2 → 
        ASPP → GlobalPool → Shared → Dual Heads
    
    Args:
        input_dim: Number of input features
        num_classes: Number of classification classes (default: 4)
        dropout: Dropout rate (default: 0.3)
        base_channels: Number of channels in first conv layer (default: 32)
    """
    
    def __init__(self, input_dim, num_classes=4, dropout=0.1, base_channels=32):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Input projection: expand features to channels
        # Shape: (batch, input_dim) -> (batch, base_channels, input_dim)
        self.input_expand = nn.Sequential(
            nn.Linear(input_dim, input_dim * base_channels),
            nn.ReLU()
        )
        self.base_channels = base_channels
        
        # Alternative: Use Conv1d to project
        # Treats input as (batch, 1, input_dim) and expands channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale conv blocks with MaxPooling
        self.ms_block1 = MultiScaleConvBlock(base_channels, base_channels * 2, dropout)  # -> 64 channels
        self.ms_block2 = MultiScaleConvBlock(base_channels * 2, base_channels * 2, dropout)  # -> 64 channels
        
        # ASPP module with dilated convolutions
        self.aspp = ASPPConvModule(base_channels * 2, base_channels * 2)
        
        # Global pooling to aggregate spatial dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Also add global max pooling for complementary features
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Shared representation (after pooling: 64*2 = 128 from avg+max pool)
        self.shared = nn.Sequential(
            nn.Linear(base_channels * 2 * 2, 64),  # 128 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape for Conv1d: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Input convolution
        x = self.input_conv(x)  # (batch, base_channels, features)
        
        # Multi-scale blocks with MaxPooling
        x = self.ms_block1(x)  # (batch, 64, features//2)
        x = self.ms_block2(x)  # (batch, 64, features//4)
        
        # ASPP
        x = self.aspp(x)  # (batch, 64, features//4)
        
        # Global pooling (both avg and max)
        avg_pool = self.global_pool(x).squeeze(-1)  # (batch, 64)
        max_pool = self.global_max_pool(x).squeeze(-1)  # (batch, 64)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 128)
        
        # Shared layers
        embedding = self.shared(pooled)
        
        # Task-specific outputs
        reg_out = self.reg_head(embedding)
        cls_out = self.cls_head(embedding)
        
        return reg_out, cls_out
    
    def get_embedding(self, x):
        """Get the shared embedding representation."""
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        x = self.ms_block1(x)
        x = self.ms_block2(x)
        x = self.aspp(x)
        avg_pool = self.global_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.shared(pooled)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(input_dim, num_classes=4, dropout=0.1, device='cpu', use_conv=False):
    """
    Factory function to create MSNet1D model.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of classification classes
        dropout: Dropout rate
        device: Device to place model on
        use_conv: If True, use Conv1D+MaxPool version; else use Linear version
    
    Returns:
        MSNet1D or MSNet1D_Conv model on specified device
    """
    if use_conv:
        model = MSNet1D_Conv(input_dim, num_classes, dropout).to(device)
        model_type = "MSNet1D_Conv (with MaxPooling)"
    else:
        model = MSNet1D(input_dim, num_classes, dropout).to(device)
        model_type = "MSNet1D (Linear)"
    
    print(f"✓ Model created: {model_type}")
    print(f"  Parameters: {count_parameters(model):,}")
    return model


def compare_architectures(input_dim, num_classes=4):
    """Print comparison of both architectures."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    model_linear = MSNet1D(input_dim, num_classes)
    model_conv = MSNet1D_Conv(input_dim, num_classes)
    
    print(f"\n{'Model':<30} {'Parameters':>15}")
    print("-" * 50)
    print(f"{'MSNet1D (Linear)':<30} {count_parameters(model_linear):>15,}")
    print(f"{'MSNet1D_Conv (MaxPool)':<30} {count_parameters(model_conv):>15,}")
    
    print("\n Key Differences:")
    print("  MSNet1D (Linear):")
    print("    - Uses fully connected layers")
    print("    - No spatial hierarchy")
    print("    - Faster training, fewer params")
    print("    - Better for small datasets")
    
    print("\n  MSNet1D_Conv (MaxPool):")
    print("    - Uses Conv1D + MaxPooling")
    print("    - Hierarchical feature extraction")
    print("    - Dilated convolutions in ASPP")
    print("    - Closer to original MSNet paper")
    print("    - May capture local patterns better")
    
    return model_linear, model_conv