"""
ablation_study.py
=================
Comprehensive ablation study for LOS prediction.

Experiments:
1. Full MSNet1D (all surgery vitals)
2. Full MSNet1D (last hour vitals)
3. MSNet1D without ASPP
4. MSNet1D without MSBlocks (simple MLP backbone)
5. Plain MLP baseline
6. MSNet1D Linear vs Conv (MaxPool)
7. Gradient Boosting baseline

Usage:
    python ablation_study.py --all          # Run all experiments
    python ablation_study.py --quick        # Run quick subset
    python ablation_study.py --exp full_model no_aspp gradient_boosting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import os
import argparse
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Data paths - UPDATE THESE FOR YOUR ENVIRONMENT
    CLINICAL_PATH = "/sfs/ceph/standard/ds7200-apt4c/vdbd_data/physionet_download/files/vitaldb/1.0.0/clinical_data.csv"
    VITALS_ALL_MEAN_PATH = "/sfs/ceph/standard/ds7200-apt4c/vdbd_data/intermediate_data/all_mean_features"
    VITALS_LAST_HOUR_PATH = "/sfs/ceph/standard/ds7200-apt4c/vdbd_data/intermediate_data/last_hour_mean_features"
    
    # Training - OPTIMIZED based on training ablation study
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 150
    EARLY_STOPPING_PATIENCE = 25
    
    # Optimal hyperparameters from ablation:
    # - Dropout 0.3: lowest val_loss (1.258), best stability (0.008), good gap (0.21)
    # - Loss weight 40/60: better than 60/40 (lower loss, smaller gap)
    # - Huber+Focal: confirmed as best loss combination
    DROPOUT = 0.3
    REG_WEIGHT = 0.4  # 40% regression, 60% classification
    CLS_WEIGHT = 0.6
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class MultiScaleBlock(nn.Module):
    """Multi-scale block with 3 parallel paths of different depths."""
    
    def __init__(self, in_features, out_features, dropout=0.3):
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
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        # Combine paths
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
    """Atrous Spatial Pyramid Pooling for tabular data."""
    
    def __init__(self, in_features, out_features, dropout=0.3):
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
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        combined = torch.cat([
            self.branch1(x), self.branch2(x),
            self.branch3(x), self.branch4(x)
        ], dim=1)
        return self.combine(combined)


# -----------------------------------------------------------------------------
# 1. Full MSNet1D (Linear version) - BASELINE
# -----------------------------------------------------------------------------
class MSNet1D_Full(nn.Module):
    """Full MSNet1D with MSBlocks + ASPP."""
    
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
        self.aspp = ASPPModule(64, 64, dropout)
        
        self.shared = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.reg_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cls_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes))
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.ms_block1(x)
        x = self.ms_block2(x)
        x = self.aspp(x)
        emb = self.shared(x)
        return self.reg_head(emb), self.cls_head(emb)


# -----------------------------------------------------------------------------
# 2. MSNet1D without ASPP
# -----------------------------------------------------------------------------
class MSNet1D_NoASPP(nn.Module):
    """MSNet1D without ASPP module."""
    
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
        # NO ASPP
        
        self.shared = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.reg_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cls_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes))
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.ms_block1(x)
        x = self.ms_block2(x)
        # Skip ASPP
        emb = self.shared(x)
        return self.reg_head(emb), self.cls_head(emb)


# -----------------------------------------------------------------------------
# 3. MSNet1D without MSBlocks (MLP backbone)
# -----------------------------------------------------------------------------
class MSNet1D_NoMSBlocks(nn.Module):
    """MSNet1D with simple MLP instead of MSBlocks."""
    
    def __init__(self, input_dim, num_classes=4, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple MLP instead of MSBlocks
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.aspp = ASPPModule(64, 64, dropout)
        
        self.shared = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.reg_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cls_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes))
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.mlp(x)  # MLP instead of MSBlocks
        x = self.aspp(x)
        emb = self.shared(x)
        return self.reg_head(emb), self.cls_head(emb)


# -----------------------------------------------------------------------------
# 4. Plain MLP (no MSBlocks, no ASPP)
# -----------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    """Simple MLP baseline - no MSBlocks, no ASPP."""
    
    def __init__(self, input_dim, num_classes=4, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.reg_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cls_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes))
    
    def forward(self, x):
        emb = self.encoder(x)
        return self.reg_head(emb), self.cls_head(emb)


# -----------------------------------------------------------------------------
# 5. MSNet1D Conv (MaxPool version)
# -----------------------------------------------------------------------------
class MultiScaleConvBlock1D(nn.Module):
    """Multi-scale Conv1D block."""
    
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.path3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.combine = nn.Sequential(
            nn.Conv1d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ) if in_channels != out_channels else nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        
        combined = torch.cat([p1, p2, p3], dim=1)
        out = self.combine(combined)
        out = self.pool(out)
        
        skip = self.skip(x)
        
        if out.shape[2] != skip.shape[2]:
            min_len = min(out.shape[2], skip.shape[2])
            out = out[:, :, :min_len]
            skip = skip[:, :, :min_len]
        
        return out + skip


class MSNet1D_Conv(nn.Module):
    """MSNet1D with Conv1D + MaxPool (treats features as 1D sequence)."""
    
    def __init__(self, input_dim, num_classes=4, dropout=0.3):
        super().__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.block1 = MultiScaleConvBlock1D(32, 64, dropout)
        self.block2 = MultiScaleConvBlock1D(64, 64, dropout)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.shared = nn.Sequential(
            nn.Linear(64 * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.reg_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cls_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes))
    
    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        x = self.input_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        emb = self.shared(pooled)
        
        return self.reg_head(emb), self.cls_head(emb)


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(config, vitals_path):
    """Load and merge clinical + vitals data."""
    
    print(f"\n  Loading clinical data: {config.CLINICAL_PATH}")
    clinical_df = pd.read_csv(config.CLINICAL_PATH)
    
    print(f"  Loading vitals data: {vitals_path}")
    if os.path.isdir(vitals_path):
        csv_files = [f for f in os.listdir(vitals_path) if f.endswith('.csv') and not f.startswith('_') and not f.startswith('.')]
        if csv_files:
            vitals_df = pd.read_csv(os.path.join(vitals_path, csv_files[0]))
        else:
            raise FileNotFoundError(f"No CSV files found in {vitals_path}")
    else:
        vitals_df = pd.read_csv(vitals_path)
    
    # Standardize case_id
    if 'caseid' in clinical_df.columns:
        clinical_df['case_id'] = clinical_df['caseid'].astype(str)
    if 'case_id' in vitals_df.columns:
        vitals_df['case_id'] = vitals_df['case_id'].astype(str)
    
    # Clean numeric columns with inequality signs
    numeric_cols = ['age', 'bmi', 'height', 'weight', 'preop_hb', 'preop_plt',
                   'preop_pt', 'preop_aptt', 'preop_na', 'preop_k', 'preop_gluc',
                   'preop_alb', 'preop_ast', 'preop_alt', 'preop_bun', 'preop_cr']
    
    for col in numeric_cols:
        if col in clinical_df.columns:
            clinical_df[col] = clinical_df[col].astype(str).str.replace(r'^[><]=?', '', regex=True)
            clinical_df[col] = pd.to_numeric(clinical_df[col], errors='coerce')
    
    # Convert timestamps
    for col in ['dis', 'adm', 'opstart', 'opend']:
        if col in clinical_df.columns:
            clinical_df[col] = pd.to_numeric(clinical_df[col], errors='coerce')
    
    # Calculate LOS and operation duration
    if 'dis' in clinical_df.columns and 'opend' in clinical_df.columns:
        clinical_df['los_hours'] = (clinical_df['dis'] - clinical_df['opend']) / 3600.0
    
    if 'opstart' in clinical_df.columns and 'opend' in clinical_df.columns:
        clinical_df['operation_duration_hours'] = (clinical_df['opend'] - clinical_df['opstart']) / 3600.0
    
    # Filter valid records
    initial_count = len(clinical_df)
    clinical_df = clinical_df[
        (clinical_df['los_hours'] >= 1) &
        (clinical_df['los_hours'] <= 720) &
        (clinical_df['operation_duration_hours'] >= 0.5) &
        (clinical_df['operation_duration_hours'] <= 24) &
        clinical_df['los_hours'].notna()
    ].copy()
    print(f"  Clinical filtered: {initial_count} -> {len(clinical_df)}")
    
    # Create LOS classes (quartiles)
    q1, q2, q3 = clinical_df['los_hours'].quantile([0.25, 0.5, 0.75])
    clinical_df['los_class'] = pd.cut(
        clinical_df['los_hours'],
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # Merge
    merged_df = clinical_df.merge(vitals_df, on='case_id', how='inner')
    print(f"  Merged: {len(merged_df)} rows")
    print(f"  LOS quartiles: Q1={q1:.1f}h, Q2={q2:.1f}h, Q3={q3:.1f}h")
    
    return merged_df


def engineer_features(df):
    """Feature engineering: target encoding, interactions, transforms."""
    
    print("\n  Feature Engineering:")
    
    # Target encoding for categorical variables
    categorical_cols = ['department', 'optype', 'asa', 'approach', 'position', 'ane_type']
    global_mean = df['los_hours'].mean()
    smoothing = 10
    
    for col in categorical_cols:
        if col in df.columns:
            stats = df.groupby(col)['los_hours'].agg(['mean', 'count'])
            smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            
            # Create BOTH features to match ablation_original_arch.py
            df[f'{col}_target_enc'] = df[col].map(smoothed_mean)
            df[f'{col}_los_mean'] = df[col].map(stats['mean'])
            
            print(f"    ✓ {col}: {df[col].nunique()} categories -> target encoded")
    
    # Numeric encodings
    if 'asa' in df.columns:
        df['asa_numeric'] = df['asa'].astype(str).str.extract(r'(\d)').astype(float)
    
    if 'sex' in df.columns:
        df['sex_male'] = df['sex'].astype(str).str.upper().isin(['M', 'MALE', '1']).astype(float)
    
    if 'emop' in df.columns:
        df['emergency'] = df['emop'].astype(str).isin(['1', 'Y', 'Yes', 'yes']).astype(float)
    
    # Interaction features
    if 'age' in df.columns and 'asa_numeric' in df.columns:
        df['age_x_asa'] = df['age'] * df['asa_numeric']
    
    if 'operation_duration_hours' in df.columns and 'asa_numeric' in df.columns:
        df['opdur_x_asa'] = df['operation_duration_hours'] * df['asa_numeric']
    
    if 'age' in df.columns and 'operation_duration_hours' in df.columns:
        df['age_x_opdur'] = df['age'] * df['operation_duration_hours']
    
    # Nonlinear transformations
    if 'bmi' in df.columns:
        df['bmi_squared'] = df['bmi'] ** 2
        df['bmi_extreme'] = ((df['bmi'] < 18.5) | (df['bmi'] > 30)).astype(float)
    
    if 'operation_duration_hours' in df.columns:
        df['opdur_log'] = np.log1p(df['operation_duration_hours'])
        df['opdur_squared'] = df['operation_duration_hours'] ** 2
    
    if 'age' in df.columns:
        df['age_squared'] = df['age'] ** 2
        df['age_log'] = np.log1p(df['age'])
    
    return df


def get_feature_columns(df):
    """Get numeric feature columns (excluding targets and IDs)."""
    
    exclude = ['case_id', 'caseid', 'subjectid', 'los_hours', 'los_class',
               'dis', 'adm', 'casestart', 'caseend', 'anestart', 'aneend',
               'opstart', 'opend',
               'department', 'optype', 'asa', 'approach', 'position', 'ane_type',
               'sex', 'emop', 'dx', 'opname', 'icu_days', 'death_inhosp']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    
    return feature_cols


def prepare_data(config, vitals_path):
    """Full data preparation pipeline."""
    
    df = load_data(config, vitals_path)
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    
    print(f"\n  Total features: {len(feature_cols)}")
    
    # Split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=config.SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config.SEED)
    
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Extract arrays
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    
    y_train = train_df['los_hours'].values.astype(np.float32)
    y_val = val_df['los_hours'].values.astype(np.float32)
    y_test = test_df['los_hours'].values.astype(np.float32)
    
    y_train_cls = train_df['los_class'].values.astype(np.int64)
    y_val_cls = val_df['los_class'].values.astype(np.int64)
    y_test_cls = test_df['los_class'].values.astype(np.int64)
    
    # Handle NaNs
    for i in range(X_train.shape[1]):
        col_median = np.nanmedian(X_train[:, i])
        if np.isnan(col_median):
            col_median = 0.0
        X_train[np.isnan(X_train[:, i]), i] = col_median
        X_val[np.isnan(X_val[:, i]), i] = col_median
        X_test[np.isnan(X_test[:, i]), i] = col_median
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_cls': y_train_cls, 'y_val_cls': y_val_cls, 'y_test_cls': y_test_cls,
        'scaler': scaler,
        'num_features': len(feature_cols),
        'feature_cols': feature_cols
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_neural_network(model, data, config, use_dual_head=True):
    """Train neural network with early stopping."""
    
    device = config.DEVICE
    model = model.to(device)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.FloatTensor(data['y_train']),
        torch.LongTensor(data['y_train_cls'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.FloatTensor(data['y_val']),
        torch.LongTensor(data['y_val_cls'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    huber = nn.HuberLoss(delta=1.0)
    focal = FocalLoss(gamma=2.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 0
    
    for epoch in range(config.MAX_EPOCHS):
        # Train
        model.train()
        for X, y_reg, y_cls in train_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            
            optimizer.zero_grad()
            
            if use_dual_head:
                reg_out, cls_out = model(X)
                # Use optimized weights: 40% regression, 60% classification
                loss = config.REG_WEIGHT * huber(reg_out.squeeze(), y_reg) + config.CLS_WEIGHT * focal(cls_out, y_cls)
            else:
                reg_out = model(X)
                loss = huber(reg_out.squeeze(), y_reg)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y_reg, y_cls in val_loader:
                X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
                
                if use_dual_head:
                    reg_out, cls_out = model(X)
                    loss = config.REG_WEIGHT * huber(reg_out.squeeze(), y_reg) + config.CLS_WEIGHT * focal(cls_out, y_cls)
                else:
                    reg_out = model(X)
                    loss = huber(reg_out.squeeze(), y_reg)
                
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if patience >= config.EARLY_STOPPING_PATIENCE:
            break
    
    model.load_state_dict(best_state)
    return model


def train_gradient_boosting(data, config):
    """Train Gradient Boosting Regressor."""
    
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=config.SEED,
        validation_fraction=0.1,
        n_iter_no_change=50,
        verbose=0
    )
    
    model.fit(data['X_train'], data['y_train'])
    return model


def evaluate_model(model, data, config, is_sklearn=False, use_dual_head=True):
    """Evaluate model on test set."""
    
    if is_sklearn:
        preds = model.predict(data['X_test'])
        cls_preds = None
    else:
        device = config.DEVICE
        model = model.to(device)
        model.eval()
        
        X_test = torch.FloatTensor(data['X_test']).to(device)
        
        with torch.no_grad():
            if use_dual_head:
                reg_out, cls_out = model(X_test)
                preds = reg_out.cpu().numpy().flatten()
                cls_preds = cls_out.argmax(1).cpu().numpy()
            else:
                reg_out = model(X_test)
                preds = reg_out.cpu().numpy().flatten()
                cls_preds = None
    
    y_test = data['y_test']
    
    results = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'mae_days': mean_absolute_error(y_test, preds) / 24.0,
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'rmse_days': np.sqrt(mean_squared_error(y_test, preds)) / 24.0
    }
    
    if cls_preds is not None:
        results['accuracy'] = accuracy_score(data['y_test_cls'], cls_preds)
    
    return results


# =============================================================================
# EXPERIMENTS DEFINITION
# =============================================================================

EXPERIMENTS = {
    # Full model variants
    "full_model": {
        "description": "Full MSNet1D (MSBlocks + ASPP) - all surgery vitals",
        "model_class": MSNet1D_Full,
        "vitals": "all_mean",
        "use_dual_head": True,
        "is_sklearn": False,
    },
    "full_model_last_hour": {
        "description": "Full MSNet1D (MSBlocks + ASPP) - last hour vitals",
        "model_class": MSNet1D_Full,
        "vitals": "last_hour",
        "use_dual_head": True,
        "is_sklearn": False,
    },
    
    # Architecture ablations
    "no_aspp": {
        "description": "MSNet1D without ASPP",
        "model_class": MSNet1D_NoASPP,
        "vitals": "all_mean",
        "use_dual_head": True,
        "is_sklearn": False,
    },
    "no_msblocks": {
        "description": "MSNet1D without MSBlocks (MLP backbone + ASPP)",
        "model_class": MSNet1D_NoMSBlocks,
        "vitals": "all_mean",
        "use_dual_head": True,
        "is_sklearn": False,
    },
    "simple_mlp": {
        "description": "Plain MLP (no MSBlocks, no ASPP)",
        "model_class": SimpleMLP,
        "vitals": "all_mean",
        "use_dual_head": True,
        "is_sklearn": False,
    },
    
    # Conv vs Linear
    "conv_maxpool": {
        "description": "MSNet1D with Conv1D + MaxPool",
        "model_class": MSNet1D_Conv,
        "vitals": "all_mean",
        "use_dual_head": True,
        "is_sklearn": False,
    },
    
    # Gradient Boosting baseline
    "gradient_boosting": {
        "description": "Gradient Boosting (sklearn)",
        "model_class": None,  # Special case
        "vitals": "all_mean",
        "use_dual_head": False,
        "is_sklearn": True,
    },
}


def run_experiment(name, exp_config, config, data):
    """Run a single experiment."""
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp_config['description']}")
    print(f"  Features: {data['num_features']}")
    print(f"  Dropout: {config.DROPOUT}, Loss weights: {config.REG_WEIGHT}/{config.CLS_WEIGHT}")
    print(f"{'='*70}")
    
    # Set seed
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    if exp_config['is_sklearn']:
        # Gradient Boosting
        print(f"  Training Gradient Boosting...")
        model = train_gradient_boosting(data, config)
        n_params = "N/A (sklearn)"
    else:
        # Neural network
        model = exp_config['model_class'](
            input_dim=data['num_features'],
            num_classes=4,
            dropout=config.DROPOUT
        )
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")
        print(f"  Training neural network...")
        model = train_neural_network(model, data, config, exp_config['use_dual_head'])
    
    # Evaluate
    results = evaluate_model(
        model, data, config,
        is_sklearn=exp_config['is_sklearn'],
        use_dual_head=exp_config['use_dual_head']
    )
    
    print(f"\n  Results:")
    print(f"    R²:        {results['r2']:.4f}")
    print(f"    MAE:       {results['mae']:.1f}h ({results['mae_days']:.2f}d)")
    print(f"    RMSE:      {results['rmse']:.1f}h ({results['rmse_days']:.2f}d)")
    if 'accuracy' in results:
        print(f"    Accuracy:  {results['accuracy']:.2%}")
    
    return {
        'experiment': name,
        'description': exp_config['description'],
        'parameters': n_params,
        **results
    }


def run_all(experiments_to_run=None):
    """Run all specified experiments."""
    
    config = Config()
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY - LOS PREDICTION")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Dropout: {config.DROPOUT}")
    print(f"Loss weights: {config.REG_WEIGHT}/{config.CLS_WEIGHT} (reg/cls)")
    print(f"Seed: {config.SEED}")
    
    if experiments_to_run is None:
        experiments_to_run = list(EXPERIMENTS.keys())
    
    # Cache data by vitals path
    data_cache = {}
    results = []
    
    for name in experiments_to_run:
        if name not in EXPERIMENTS:
            print(f"\n⚠️  Unknown experiment: {name}")
            continue
            
        exp_config = EXPERIMENTS[name]
        
        # Determine vitals path
        vitals_key = exp_config['vitals']
        if vitals_key == 'all_mean':
            vitals_path = config.VITALS_ALL_MEAN_PATH
        elif vitals_key == 'last_hour':
            vitals_path = config.VITALS_LAST_HOUR_PATH
        else:
            vitals_path = config.VITALS_ALL_MEAN_PATH
        
        # Load data if not cached
        if vitals_key not in data_cache:
            print(f"\n{'='*70}")
            print(f"LOADING DATA: {vitals_key}")
            print(f"{'='*70}")
            try:
                data_cache[vitals_key] = prepare_data(config, vitals_path)
            except Exception as e:
                print(f"  ERROR loading data: {e}")
                continue
        
        data = data_cache[vitals_key]
        
        try:
            res = run_experiment(name, exp_config, config, data)
            res['num_features'] = data['num_features']
            res['vitals'] = vitals_key
            results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'experiment': name, 'error': str(e)})
    
    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ablation_results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    print(f"\n{'Experiment':<25} {'Vitals':<12} {'R²':>8} {'MAE(d)':>8} {'RMSE(d)':>8} {'Params':>12}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        if 'error' in row and pd.notna(row.get('error')):
            print(f"{row['experiment']:<25} ERROR: {row['error']}")
        else:
            params = row.get('parameters', 'N/A')
            if isinstance(params, (int, float)) and not np.isnan(params):
                params = f"{int(params):,}"
            print(f"{row['experiment']:<25} {row.get('vitals', 'N/A'):<12} "
                  f"{row['r2']:>8.4f} {row['mae_days']:>8.2f} {row['rmse_days']:>8.2f} "
                  f"{params:>12}")
    
    print("-" * 90)
    
    # Find best
    valid_results = results_df[~results_df.get('error', pd.Series([None]*len(results_df))).notna()]
    if len(valid_results) > 0 and 'r2' in valid_results.columns:
        best_idx = valid_results['r2'].idxmax()
        best = valid_results.loc[best_idx]
        print(f"\n🏆 Best: {best['experiment']} (R² = {best['r2']:.4f})")
    
    return results_df


def run_quick():
    """Run quick subset of experiments."""
    return run_all([
        "full_model",
        "no_aspp",
        "no_msblocks", 
        "simple_mlp",
        "conv_maxpool",
        "gradient_boosting",
    ])


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for LOS prediction")
    parser.add_argument("--quick", action="store_true", help="Run quick experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--exp", nargs="+", help="Run specific experiments")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable experiments:")
        for name, config in EXPERIMENTS.items():
            print(f"  {name:<25} - {config['description']}")
        exit(0)
    
    if args.all:
        run_all()
    elif args.exp:
        run_all(args.exp)
    elif args.quick:
        run_quick()
    else:
        print("Usage:")
        print("  python ablation_study.py --quick    # Run quick experiments")
        print("  python ablation_study.py --all      # Run all experiments")
        print("  python ablation_study.py --exp full_model no_aspp gradient_boosting")
        print("  python ablation_study.py --list     # List available experiments")
        print("\nRunning quick experiments by default...")
        run_quick()