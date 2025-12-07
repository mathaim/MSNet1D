"""
train.py
========
Training functions for LOS prediction models.

Contains:
- train_gradient_boosting: Train sklearn GradientBoostingRegressor
- train_neural_network: Train MSNet1D with early stopping
- evaluate_model: Compute regression metrics
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model import FocalLoss


def train_gradient_boosting(X_train, y_train, X_val, y_val, config):
    """
    Train Gradient Boosting Regressor.
    
    Args:
        X_train: Training features (scaled)
        y_train: Training targets
        X_val: Validation features (scaled)
        y_val: Validation targets
        config: Configuration object
    
    Returns:
        Fitted GradientBoostingRegressor model
    """
    print("\n" + "=" * 60)
    print("TRAINING GRADIENT BOOSTING")
    print("=" * 60)
    
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
    
    print("  Training...")
    model.fit(X_train, y_train)
    
    # Evaluate
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    
    train_r2 = r2_score(y_train, pred_train)
    val_r2 = r2_score(y_val, pred_val)
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Val R²:   {val_r2:.4f}")
    print(f"  Trees used: {model.n_estimators_}")
    
    return model


def train_neural_network(X_train, y_train, y_train_class,
                        X_val, y_val, y_val_class,
                        config, device, use_conv=False):
    """
    Train MSNet1D neural network with early stopping.
    
    Uses:
    - Huber loss for regression (robust to outliers)
    - Focal loss for classification (handles imbalance)
    - Weighted sampling for class balance
    - Learning rate scheduling
    - Gradient clipping
    
    Args:
        X_train, y_train, y_train_class: Training data
        X_val, y_val, y_val_class: Validation data
        config: Configuration object
        device: torch device
        use_conv: If True, use Conv1D+MaxPool architecture
    
    Returns:
        Trained MSNet1D model (best checkpoint), history dict
    """
    from model import MSNet1D, MSNet1D_Conv
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.LongTensor(y_train_class.astype(int))
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.LongTensor(y_val_class.astype(int))
    )
    
    # Weighted sampler for class imbalance
    class_counts = np.bincount(y_train_class.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train_class.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # Create model (choose architecture based on use_conv)
    if use_conv:
        model = MSNet1D_Conv(
            input_dim=X_train.shape[1],
            num_classes=4,
            dropout=config.DROPOUT
        ).to(device)
        arch_name = "MSNet1D_Conv (MaxPool)"
    else:
        model = MSNet1D(
            input_dim=X_train.shape[1],
            num_classes=4,
            dropout=config.DROPOUT
        ).to(device)
        arch_name = "MSNet1D (Linear)"
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {arch_name}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device}")
    
    # Loss functions
    huber_loss = nn.HuberLoss(delta=1.0)
    focal_loss = FocalLoss(gamma=2.0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    print(f"\n  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val R²':>8}  {'LR':>10}")
    print("  " + "-" * 52)
    
    for epoch in range(config.MAX_EPOCHS):
        # ========== TRAINING ==========
        model.train()
        train_loss = 0
        
        for X_batch, y_reg, y_cls in train_loader:
            X_batch = X_batch.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)
            
            optimizer.zero_grad()
            
            reg_out, cls_out = model(X_batch)
            
            # Multi-task loss
            loss_reg = huber_loss(reg_out.squeeze(), y_reg)
            loss_cls = focal_loss(cls_out, y_cls)
            loss = 0.6 * loss_reg + 0.4 * loss_cls
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_reg, y_cls in val_loader:
                X_batch = X_batch.to(device)
                y_reg = y_reg.to(device)
                y_cls = y_cls.to(device)
                
                reg_out, cls_out = model(X_batch)
                
                loss_reg = huber_loss(reg_out.squeeze(), y_reg)
                loss_cls = focal_loss(cls_out, y_cls)
                val_loss += (0.6 * loss_reg + 0.4 * loss_cls).item()
                
                val_preds.extend(reg_out.cpu().numpy().flatten())
                val_targets.extend(y_reg.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_r2 = r2_score(val_targets, val_preds)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0 or patience_counter == 0:
            marker = " *" if patience_counter == 0 else ""
            print(f"  {epoch+1:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  "
                  f"{val_r2:>8.4f}  {current_lr:>10.6f}{marker}")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n  ✓ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"  ✓ Loaded best model (val_loss: {best_val_loss:.4f})")
    
    return model, history


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate regression model performance.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name for display
    
    Returns:
        Dictionary with metrics
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'name': model_name,
        'r2': r2,
        'mae': mae,
        'mae_days': mae / 24,
        'rmse': rmse,
        'rmse_days': rmse / 24
    }


def print_metrics(metrics):
    """Print formatted metrics."""
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAE:  {metrics['mae']:.1f}h ({metrics['mae_days']:.2f}d)")
    print(f"  RMSE: {metrics['rmse']:.1f}h ({metrics['rmse_days']:.2f}d)")


def predict_neural_network(model, X, device):
    """
    Generate predictions from neural network.
    
    Args:
        model: Trained MSNet1D model
        X: Input features (numpy array, scaled)
        device: torch device
    
    Returns:
        Numpy array of predictions
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        reg_out, _ = model(X_tensor)
        return reg_out.cpu().numpy().flatten()