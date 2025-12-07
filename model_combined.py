"""
model.py
========
Neural network architectures for LOS prediction.

Contains:
- FocalLoss: Loss function for imbalanced classification
- MSNet1D_Vitals: Conv1D network for vital sign time-series
- ClinicalEncoder: MLP for clinical features
- MSNet1D_Combined: Combined model (vitals + clinical)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    USE_DUAL_HEAD = True      # False = regression only
    USE_SEPARATE_ENCODERS = True  # False = concatenate inputs
    USE_MULTISCALE = True     # False = single kernel size
    USE_ASPP = True           # False = skip ASPP module
    USE_VITALS = True         # False = clinical only baseline
    
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# =============================================================================
# VITAL SIGNS ENCODER (Conv1D for time-series)
# =============================================================================

class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale 1D convolution block for vital sign time-series.
    
    Uses parallel convolutions with different kernel sizes to capture
    patterns at multiple temporal scales, plus MaxPooling for downsampling.
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        
        # Path 1: Small kernel (short-term patterns, ~seconds)
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Path 2: Medium kernel (medium-term patterns, ~tens of seconds)
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Path 3: Large kernel (long-term patterns, ~minutes)
        self.path3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MaxPool for temporal downsampling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Combine paths
        self.combine = nn.Sequential(
            nn.Conv1d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ) if in_channels != out_channels else nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        
        # Parallel paths
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        
        # Concatenate and combine
        combined = torch.cat([p1, p2, p3], dim=1)
        out = self.combine(combined)
        out = self.pool(out)
        
        # Skip connection
        skip = self.skip(x)
        
        # Match dimensions if needed
        if out.shape[2] != skip.shape[2]:
            min_len = min(out.shape[2], skip.shape[2])
            out = out[:, :, :min_len]
            skip = skip[:, :, :min_len]
        
        return out + skip


class ASPPModule1D(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 1D time-series.
    
    Uses dilated convolutions to capture multi-scale temporal context
    without losing resolution.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        hidden = in_channels // 4
        
        # Branch 1: No dilation (local context)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 2: Small dilation (short-range context)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 3: Medium dilation (medium-range context)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 4: Large dilation (long-range context)
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Branch 5: Global context
        self.branch5_pool = nn.AdaptiveAvgPool1d(1)
        self.branch5_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        
        # Combine all branches
        self.combine = nn.Sequential(
            nn.Conv1d(hidden * 5, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        size = x.shape[2]
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Global pooling branch
        b5 = self.branch5_pool(x)
        b5 = self.branch5_conv(b5)
        b5 = F.interpolate(b5, size=size, mode='nearest')
        
        # Concatenate all branches
        combined = torch.cat([b1, b2, b3, b4, b5], dim=1)
        
        return self.combine(combined)


class VitalsEncoder(nn.Module):
    """
    Encoder for vital sign time-series using Conv1D + MaxPool + ASPP.
    
    Input: (batch, num_signals, time_steps)
        e.g., (32, 20, 360) for 20 vital signals over 360 time points (60 min at 0.1Hz)
    
    Output: (batch, embedding_dim)
    """
    
    def __init__(self, num_signals, embedding_dim=128, dropout=0.3):
        super().__init__()
        
        self.num_signals = num_signals
        
        # Initial convolution to expand channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(num_signals, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale blocks with MaxPooling
        self.block1 = MultiScaleConvBlock(64, 128, dropout)   # time/2
        self.block2 = MultiScaleConvBlock(128, 128, dropout)  # time/4
        self.block3 = MultiScaleConvBlock(128, 256, dropout)  # time/8
        
        # ASPP for multi-scale context
        self.aspp = ASPPModule1D(256, 256)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(256 * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: (batch, num_signals, time_steps)
        
        # Initial conv
        x = self.input_conv(x)
        
        # Multi-scale blocks with MaxPooling
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Global pooling (both avg and max)
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        
        # Concatenate and project
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        embedding = self.projection(pooled)
        
        return embedding


# =============================================================================
# CLINICAL FEATURES ENCODER (MLP)
# =============================================================================

class ClinicalEncoder(nn.Module):
    """
    Encoder for clinical features using MLP.
    
    Input: (batch, num_clinical_features)
    Output: (batch, embedding_dim)
    """
    
    def __init__(self, num_features, embedding_dim=64, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.encoder(x)


# =============================================================================
# COMBINED MODEL (Vitals + Clinical)
# =============================================================================

class MSNet1D_Combined(nn.Module):
    """
    Combined model for LOS prediction using both vital signs and clinical features.
    
    Architecture:
        Vital Signs (time-series) → VitalsEncoder (Conv1D + ASPP) → vitals_embedding
        Clinical Features → ClinicalEncoder (MLP) → clinical_embedding
        
        Concatenate → Fusion → Shared → Dual Heads (Regression + Classification)
    
    Args:
        num_vital_signals: Number of vital sign channels (e.g., 20)
        num_clinical_features: Number of clinical features (e.g., 50)
        num_classes: Number of LOS classes for classification (default: 4)
        vitals_embedding_dim: Dimension of vitals embedding (default: 128)
        clinical_embedding_dim: Dimension of clinical embedding (default: 64)
        dropout: Dropout rate (default: 0.3)
    
    Returns:
        Tuple of (regression_output, classification_output)
    """
    
    def __init__(self, num_vital_signals, num_clinical_features, 
                 num_classes=4, vitals_embedding_dim=128, 
                 clinical_embedding_dim=64, dropout=0.3):
        super().__init__()
        
        self.num_vital_signals = num_vital_signals
        self.num_clinical_features = num_clinical_features
        
        # Encoders
        self.vitals_encoder = VitalsEncoder(
            num_signals=num_vital_signals,
            embedding_dim=vitals_embedding_dim,
            dropout=dropout
        )
        
        self.clinical_encoder = ClinicalEncoder(
            num_features=num_clinical_features,
            embedding_dim=clinical_embedding_dim,
            dropout=dropout
        )
        
        # Fusion layer
        combined_dim = vitals_embedding_dim + clinical_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, vitals, clinical):
        """
        Args:
            vitals: (batch, num_signals, time_steps) - vital sign time-series
            clinical: (batch, num_clinical_features) - clinical features
        
        Returns:
            reg_out: (batch, 1) - LOS prediction in hours
            cls_out: (batch, num_classes) - LOS class logits
        """
        # Encode vitals (time-series through Conv1D)
        vitals_emb = self.vitals_encoder(vitals)
        
        # Encode clinical (tabular through MLP)
        clinical_emb = self.clinical_encoder(clinical)
        
        # Fuse
        combined = torch.cat([vitals_emb, clinical_emb], dim=1)
        fused = self.fusion(combined)
        
        # Shared representation
        shared = self.shared(fused)
        
        # Output heads
        reg_out = self.reg_head(shared)
        cls_out = self.cls_head(shared)
        
        return reg_out, cls_out
    
    def get_embedding(self, vitals, clinical):
        """Get the shared embedding representation."""
        vitals_emb = self.vitals_encoder(vitals)
        clinical_emb = self.clinical_encoder(clinical)
        combined = torch.cat([vitals_emb, clinical_emb], dim=1)
        fused = self.fusion(combined)
        return self.shared(fused)


# =============================================================================
# VITALS-ONLY MODEL (for comparison)
# =============================================================================

class MSNet1D_VitalsOnly(nn.Module):
    """
    Model using only vital signs (no clinical features).
    For ablation studies comparing vitals-only vs combined.
    """
    
    def __init__(self, num_vital_signals, num_classes=4, 
                 embedding_dim=128, dropout=0.3):
        super().__init__()
        
        self.vitals_encoder = VitalsEncoder(
            num_signals=num_vital_signals,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, vitals):
        emb = self.vitals_encoder(vitals)
        shared = self.shared(emb)
        return self.reg_head(shared), self.cls_head(shared)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_combined_model(num_vital_signals, num_clinical_features, 
                          num_classes=4, dropout=0.3, device='cpu'):
    """
    Factory function to create the combined model.
    """
    model = MSNet1D_Combined(
        num_vital_signals=num_vital_signals,
        num_clinical_features=num_clinical_features,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    print(f"✓ Created MSNet1D_Combined:")
    print(f"  Vital signals: {num_vital_signals}")
    print(f"  Clinical features: {num_clinical_features}")
    print(f"  Total parameters: {count_parameters(model):,}")
    
    return model


# =============================================================================
# KEEP OLD MODELS FOR BACKWARD COMPATIBILITY
# =============================================================================

# Include the original MSNet1D and MSNet1D_Conv classes here if needed
# (keeping them for comparison with aggregated features)

class MultiScaleBlock(nn.Module):
    """Multi-scale block for tabular data (Linear version)."""
    
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        
        self.path1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
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
        
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        self.combine = nn.Sequential(
            nn.Linear(out_features * 3, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        skip = self.skip(x)
        combined = torch.cat([p1, p2, p3], dim=1)
        out = self.combine(combined)
        return out + skip


class ASPPModule(nn.Module):
    """ASPP for tabular data (Linear version)."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        hidden = in_features // 4
        
        self.branch1 = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
        )
        
        self.combine = nn.Sequential(
            nn.Linear(hidden * 4, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        combined = torch.cat([
            self.branch1(x), self.branch2(x), 
            self.branch3(x), self.branch4(x)
        ], dim=1)
        return self.combine(combined)


class MSNet1D(nn.Module):
    """Original MSNet1D for tabular data (Linear version)."""
    
    def __init__(self, input_dim, num_classes=4, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.ms_block1 = MultiScaleBlock(128, 64, dropout)
        self.ms_block2 = MultiScaleBlock(64, 64, dropout)
        self.aspp = ASPPModule(64, 64)
        
        self.shared = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.ms_block1(x)
        x = self.ms_block2(x)
        x = self.aspp(x)
        embedding = self.shared(x)
        return self.reg_head(embedding), self.cls_head(embedding)