"""
Training Configuration Ablation Study

Loads data directly with pandas (no Spark needed).
Evaluates: loss weights, dropout rates, loss functions

Usage:
    python training_ablation_simple.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# Data Loading (Pandas only - no Spark)
# ==============================================================================

def load_and_preprocess_data():
    """Load and preprocess VitalDB data using pandas."""
    
    CLINICAL_PATH = '/sfs/ceph/standard/ds7200-apt4c/vdbd_data/physionet_download/files/vitaldb/1.0.0/clinical_data.csv'
    VITALS_PATH = '/sfs/ceph/standard/ds7200-apt4c/vdbd_data/intermediate_data/all_mean_features/'
    
    print("Loading data...")
    
    # Load clinical data
    clinical_df = pd.read_csv(CLINICAL_PATH)
    print(f"  Clinical: {clinical_df.shape}")
    
    # Load vitals data
    vitals_files = [f for f in os.listdir(VITALS_PATH) if f.endswith('.csv') and not f.startswith('.')]
    vitals_dfs = [pd.read_csv(os.path.join(VITALS_PATH, f)) for f in vitals_files]
    vitals_df = pd.concat(vitals_dfs, ignore_index=True)
    
    # Rename case_id to caseid
    if 'case_id' in vitals_df.columns:
        vitals_df = vitals_df.rename(columns={'case_id': 'caseid'})
    print(f"  Vitals: {vitals_df.shape}")
    
    # Merge
    df = clinical_df.merge(vitals_df, on='caseid', how='inner')
    print(f"  Merged: {df.shape}")
    
    # Compute LOS
    df['los_hours'] = (df['dis'] - df['opend']) / 3600.0
    df['los_days'] = df['los_hours'] / 24.0
    
    # Filter valid LOS
    df = df[(df['los_hours'] >= 1) & (df['los_days'] <= 30)]
    print(f"  After LOS filter: {df.shape}")
    
    # Compute quartiles for classification
    df['los_class'] = pd.qcut(df['los_hours'], q=4, labels=False, duplicates='drop')
    
    # Clean numeric columns with inequality signs (e.g., ">89")
    numeric_cols = ['age', 'bmi', 'height', 'weight', 'preop_hb', 'preop_plt', 
                    'preop_pt', 'preop_aptt', 'preop_na', 'preop_k', 'preop_gluc',
                    'preop_alb', 'preop_ast', 'preop_alt', 'preop_bun', 'preop_cr']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'^[><]=?', '', regex=True), errors='coerce')
    
    # Select feature columns (numeric only, exclude IDs and targets)
    exclude = ['caseid', 'subjectid', 'los_hours', 'los_days', 'los_class', 
               'dis', 'adm', 'opstart', 'opend', 'anestart', 'aneend',
               'casestart', 'caseend', 'opname', 'dx', 'department', 'optype',
               'asa', 'approach', 'position', 'ane_type', 'sex', 'emop']
    
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    print(f"  Features: {len(feature_cols)}")
    
    # Handle missing values
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['los_hours'].values / 24.0  # Convert to days
    y_cls = df['los_class'].values.astype(int)
    
    # Drop any remaining NaN/inf
    mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
    X = X[mask].values
    y = y[mask.values]
    y_cls = y_cls[mask.values]
    
    # Split
    X_train, X_val, y_train, y_val, y_cls_train, y_cls_val = train_test_split(
        X, y, y_cls, test_size=0.15, random_state=SEED, stratify=y_cls
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"\nData ready:")
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  LOS range: {y_train.min():.1f} - {y_train.max():.1f} days")
    
    return X_train, X_val, y_train, y_val, y_cls_train, y_cls_val


# ==============================================================================
# Model
# ==============================================================================

class MultiScaleBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.path1 = nn.Sequential(
            nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Dropout(dropout))
        self.path2 = nn.Sequential(
            nn.Linear(in_features, in_features), nn.BatchNorm1d(in_features), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Dropout(dropout))
        bottleneck = max(in_features // 2, 16)
        self.path3 = nn.Sequential(
            nn.Linear(in_features, bottleneck), nn.BatchNorm1d(bottleneck), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(bottleneck, bottleneck), nn.BatchNorm1d(bottleneck), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(bottleneck, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Dropout(dropout))
        self.combine = nn.Linear(out_features * 3, out_features)
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        return self.combine(torch.cat([self.path1(x), self.path2(x), self.path3(x)], dim=1)) + self.skip(x)


class MSNet1D_NoASPP(nn.Module):
    def __init__(self, input_dim, dropout=0.1, num_classes=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout))
        self.msblock1 = MultiScaleBlock(128, 64, dropout)
        self.msblock2 = MultiScaleBlock(64, 64, dropout)
        self.shared = nn.Sequential(
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout))
        self.reg_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cls_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes))

    def forward(self, x):
        x = self.shared(self.msblock2(self.msblock1(self.input_proj(x))))
        return self.reg_head(x), self.cls_head(x)


# ==============================================================================
# Loss Functions
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ==============================================================================
# Training
# ==============================================================================

def train_model(model, train_loader, val_loader, config, device):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    reg_criterion = nn.HuberLoss() if config['reg_loss'] == 'huber' else nn.MSELoss()
    cls_criterion = FocalLoss() if config['cls_loss'] == 'focal' else nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(150):
        # Train
        model.train()
        train_losses = []
        for X, y_reg, y_cls in train_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            optimizer.zero_grad()
            reg_out, cls_out = model(X)
            loss = config['reg_weight'] * reg_criterion(reg_out.squeeze(), y_reg) + \
                   (1 - config['reg_weight']) * cls_criterion(cls_out, y_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y_reg, y_cls in val_loader:
                X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
                reg_out, cls_out = model(X)
                loss = config['reg_weight'] * reg_criterion(reg_out.squeeze(), y_reg) + \
                       (1 - config['reg_weight']) * cls_criterion(cls_out, y_cls)
                val_losses.append(loss.item())
        
        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break
    
    metrics = {
        'final_val_loss': best_val_loss,
        'stability': np.std(history['val_loss'][-10:]) if len(history['val_loss']) >= 10 else np.std(history['val_loss']),
        'generalization_gap': np.mean(history['val_loss'][-10:]) - np.mean(history['train_loss'][-10:]) if len(history['val_loss']) >= 10 else 0,
        'epochs': len(history['train_loss'])
    }
    
    return history, metrics


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("TRAINING CONFIGURATION ABLATION")
    print("=" * 60)
    
    # Load data
    X_train, X_val, y_train, y_val, y_cls_train, y_cls_val = load_and_preprocess_data()
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor(y_cls_train))
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val), torch.LongTensor(y_cls_val))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = X_train.shape[1]
    print(f"\nUsing device: {device}")
    
    results = {}
    histories = {}
    
    # ========== Experiment 1: Loss Weights ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Loss Weights")
    print("=" * 60)
    
    for reg_weight, name in [(1.0, '100/0'), (0.8, '80/20'), (0.6, '60/40'), (0.4, '40/60'), (0.2, '20/80')]:
        print(f"\n{name}...", end=" ", flush=True)
        torch.manual_seed(SEED)
        model = MSNet1D_NoASPP(input_dim, dropout=0.1)
        cfg = {'reg_weight': reg_weight, 'reg_loss': 'huber', 'cls_loss': 'focal'}
        h, m = train_model(model, train_loader, val_loader, cfg, device)
        results[f'weight_{name}'] = m
        histories[f'weight_{name}'] = h
        print(f"val_loss={m['final_val_loss']:.4f}, stability={m['stability']:.4f}, epochs={m['epochs']}")
    
    # ========== Experiment 2: Dropout ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Dropout Rates")
    print("=" * 60)
    
    for dropout in [0.0, 0.1, 0.2, 0.3, 0.5]:
        print(f"\ndropout={dropout}...", end=" ", flush=True)
        torch.manual_seed(SEED)
        model = MSNet1D_NoASPP(input_dim, dropout=dropout)
        cfg = {'reg_weight': 0.6, 'reg_loss': 'huber', 'cls_loss': 'focal'}
        h, m = train_model(model, train_loader, val_loader, cfg, device)
        results[f'dropout_{dropout}'] = m
        histories[f'dropout_{dropout}'] = h
        print(f"val_loss={m['final_val_loss']:.4f}, stability={m['stability']:.4f}, epochs={m['epochs']}")
    
    # ========== Experiment 3: Loss Functions ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Loss Functions")
    print("=" * 60)
    
    for reg_loss, cls_loss, name in [('huber', 'focal', 'Huber+Focal'), ('mse', 'focal', 'MSE+Focal'),
                                      ('huber', 'ce', 'Huber+CE'), ('mse', 'ce', 'MSE+CE')]:
        print(f"\n{name}...", end=" ", flush=True)
        torch.manual_seed(SEED)
        model = MSNet1D_NoASPP(input_dim, dropout=0.1)
        cfg = {'reg_weight': 0.6, 'reg_loss': reg_loss, 'cls_loss': cls_loss}
        h, m = train_model(model, train_loader, val_loader, cfg, device)
        results[f'loss_{name}'] = m
        histories[f'loss_{name}'] = h
        print(f"val_loss={m['final_val_loss']:.4f}, stability={m['stability']:.4f}, epochs={m['epochs']}")
    
    # ========== Plot ==========
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss weights
    for key in [k for k in histories if k.startswith('weight_')]:
        label = key.replace('weight_', '')
        lw = 2.5 if '60/40' in key else 1.5
        axes[0, 0].plot(histories[key]['val_loss'], label=label, linewidth=lw)
    axes[0, 0].set_title('(a) Loss Weight: Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Val Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Train vs Val for 60/40
    h = histories.get('weight_60/40', list(histories.values())[0])
    axes[0, 1].plot(h['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(h['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 1].fill_between(range(len(h['train_loss'])), h['train_loss'], h['val_loss'], alpha=0.2)
    axes[0, 1].set_title('(b) 60/40: Train vs Val', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Dropout
    for key in [k for k in histories if k.startswith('dropout_')]:
        label = key.replace('dropout_', '')
        lw = 2.5 if '0.1' in key else 1.5
        axes[0, 2].plot(histories[key]['val_loss'], label=label, linewidth=lw)
    axes[0, 2].set_title('(c) Dropout: Validation Loss', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Loss functions
    for key in [k for k in histories if k.startswith('loss_')]:
        label = key.replace('loss_', '')
        lw = 2.5 if 'Huber+Focal' in key else 1.5
        axes[1, 0].plot(histories[key]['val_loss'], label=label, linewidth=lw)
    axes[1, 0].set_title('(d) Loss Function: Validation Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Smoothed
    for key in [k for k in histories if k.startswith('loss_')]:
        label = key.replace('loss_', '')
        smoothed = uniform_filter1d(histories[key]['val_loss'], size=5)
        axes[1, 1].plot(smoothed, label=label, linewidth=1.5)
    axes[1, 1].set_title('(e) Loss Functions: Smoothed', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # LR schedule
    axes[1, 2].plot(h['lr'], 'g-', linewidth=2)
    axes[1, 2].set_title('(f) Learning Rate Schedule', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Saved training_curves.png")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('training_ablation_results.csv')
    print("Saved training_ablation_results.csv")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string())
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()