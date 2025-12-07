"""
ensemble.py
===========
Ensemble methods for combining Gradient Boosting and Neural Network.

Methods:
1. Simple Average: (pred_gb + pred_nn) / 2
2. Weighted Average: w1 * pred_gb + w2 * pred_nn (learned weights)
3. Stacking: Ridge regression meta-learner

Reference:
    Shwartz-Ziv & Armon, "Tabular Data: Deep Learning is Not All You Need", 2022
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


class EnsemblePredictor:
    """
    Ensemble combining Gradient Boosting and Neural Network predictions.
    
    Usage:
        ensemble = EnsemblePredictor(gb_model, nn_model, scaler, device)
        
        # Simple average
        pred = ensemble.simple_average(X)
        
        # Weighted average (learns weights on validation set)
        ensemble.fit_weights(X_val, y_val)
        pred = ensemble.weighted_average(X)
        
        # Stacking (learns meta-model on validation set)
        ensemble.fit_stacking(X_val, y_val)
        pred = ensemble.stacking_predict(X)
    
    Args:
        gb_model: Trained GradientBoostingRegressor
        nn_model: Trained MSNet1D model
        scaler: Fitted StandardScaler
        device: torch device for neural network
    """
    
    def __init__(self, gb_model, nn_model, scaler, device='cpu'):
        self.gb_model = gb_model
        self.nn_model = nn_model
        self.scaler = scaler
        self.device = device
        
        # Learned ensemble parameters
        self.weights = None
        self.meta_learner = None
    
    def _get_base_predictions(self, X):
        """
        Get predictions from both base models.
        
        Args:
            X: Raw features (unscaled)
        
        Returns:
            Tuple of (gb_predictions, nn_predictions)
        """
        X_scaled = self.scaler.transform(X)
        
        # Gradient Boosting prediction
        pred_gb = self.gb_model.predict(X_scaled)
        
        # Neural Network prediction
        self.nn_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            pred_nn, _ = self.nn_model(X_tensor)
            pred_nn = pred_nn.cpu().numpy().flatten()
        
        return pred_gb, pred_nn
    
    def simple_average(self, X):
        """
        Simple average of base model predictions.
        
        Formula: (pred_gb + pred_nn) / 2
        
        Args:
            X: Raw features (unscaled)
        
        Returns:
            Averaged predictions
        """
        pred_gb, pred_nn = self._get_base_predictions(X)
        return (pred_gb + pred_nn) / 2
    
    def fit_weights(self, X_val, y_val):
        """
        Learn optimal weights on validation set via grid search.
        
        Searches over weight combinations in [0, 1] with step 0.05.
        Constraint: w1 + w2 = 1
        
        Args:
            X_val: Validation features (unscaled)
            y_val: Validation targets
        
        Returns:
            Tuple of (best_weights, best_r2)
        """
        print("\n📊 Learning ensemble weights...")
        
        pred_gb, pred_nn = self._get_base_predictions(X_val)
        
        best_r2 = -np.inf
        best_weights = (0.5, 0.5)
        
        # Grid search
        for w1 in np.arange(0, 1.05, 0.05):
            w2 = 1 - w1
            pred_ensemble = w1 * pred_gb + w2 * pred_nn
            r2 = r2_score(y_val, pred_ensemble)
            
            if r2 > best_r2:
                best_r2 = r2
                best_weights = (w1, w2)
        
        self.weights = best_weights
        
        print(f"  Weights: GB={best_weights[0]:.2f}, NN={best_weights[1]:.2f}")
        print(f"  Val R²:  {best_r2:.4f}")
        
        return best_weights, best_r2
    
    def weighted_average(self, X):
        """
        Weighted average using learned weights.
        
        Must call fit_weights() first.
        
        Args:
            X: Raw features (unscaled)
        
        Returns:
            Weighted predictions
        """
        if self.weights is None:
            raise ValueError("Must call fit_weights() first")
        
        pred_gb, pred_nn = self._get_base_predictions(X)
        return self.weights[0] * pred_gb + self.weights[1] * pred_nn
    
    def fit_stacking(self, X_val, y_val):
        """
        Train stacking meta-learner on validation predictions.
        
        Uses Ridge regression as the meta-learner for:
        - Regularization to prevent overfitting
        - Interpretable coefficients
        
        Args:
            X_val: Validation features (unscaled)
            y_val: Validation targets
        
        Returns:
            Validation R² of meta-learner
        """
        print("\n Training stacking meta-learner...")
        
        pred_gb, pred_nn = self._get_base_predictions(X_val)
        
        # Stack predictions as meta-features
        meta_features = np.column_stack([pred_gb, pred_nn])
        
        # Train Ridge regression
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(meta_features, y_val)
        
        # Evaluate
        meta_pred = self.meta_learner.predict(meta_features)
        val_r2 = r2_score(y_val, meta_pred)
        
        print(f"  Coefficients: GB={self.meta_learner.coef_[0]:.3f}, "
              f"NN={self.meta_learner.coef_[1]:.3f}")
        print(f"  Intercept:    {self.meta_learner.intercept_:.3f}")
        print(f"  Val R²:       {val_r2:.4f}")
        
        return val_r2
    
    def stacking_predict(self, X):
        """
        Predict using stacking meta-learner.
        
        Must call fit_stacking() first.
        
        Args:
            X: Raw features (unscaled)
        
        Returns:
            Stacking predictions
        """
        if self.meta_learner is None:
            raise ValueError("Must call fit_stacking() first")
        
        pred_gb, pred_nn = self._get_base_predictions(X)
        meta_features = np.column_stack([pred_gb, pred_nn])
        
        return self.meta_learner.predict(meta_features)
    
    def get_all_predictions(self, X, y=None):
        """
        Get predictions from all ensemble methods.
        
        Args:
            X: Raw features (unscaled)
            y: Optional true values for evaluation
        
        Returns:
            Dictionary of predictions (and metrics if y provided)
        """
        pred_gb, pred_nn = self._get_base_predictions(X)
        
        results = {
            'pred_gb': pred_gb,
            'pred_nn': pred_nn,
            'pred_simple': (pred_gb + pred_nn) / 2
        }
        
        if self.weights is not None:
            results['pred_weighted'] = (
                self.weights[0] * pred_gb + self.weights[1] * pred_nn
            )
        
        if self.meta_learner is not None:
            meta_features = np.column_stack([pred_gb, pred_nn])
            results['pred_stacking'] = self.meta_learner.predict(meta_features)
        
        if y is not None:
            results['metrics'] = {}
            for key in results:
                if key.startswith('pred_'):
                    name = key.replace('pred_', '')
                    results['metrics'][name] = {
                        'r2': r2_score(y, results[key]),
                        'mae': np.mean(np.abs(y - results[key])),
                        'rmse': np.sqrt(np.mean((y - results[key])**2))
                    }
        
        return results


def analyze_ensemble(results, y_test):
    """
    Analyze and print ensemble results.
    
    Args:
        results: Dictionary of model results with 'r2' key
        y_test: Test targets (for context)
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE ANALYSIS")
    print("=" * 80)
    
    # Find best single model
    single_models = {k: v for k, v in results.items() 
                    if 'Ensemble' not in k}
    best_single = max(single_models, key=lambda k: single_models[k]['r2'])
    best_single_r2 = single_models[best_single]['r2']
    
    # Find best ensemble
    ensemble_models = {k: v for k, v in results.items() 
                      if 'Ensemble' in k}
    if ensemble_models:
        best_ensemble = max(ensemble_models, key=lambda k: ensemble_models[k]['r2'])
        best_ensemble_r2 = ensemble_models[best_ensemble]['r2']
        
        improvement = best_ensemble_r2 - best_single_r2
        
        print(f"\n  Best single model: {best_single}")
        print(f"    R² = {best_single_r2:.4f}")
        
        print(f"\n  Best ensemble: {best_ensemble}")
        print(f"    R² = {best_ensemble_r2:.4f}")
        
        if improvement > 0:
            print(f"\n  ✅ Ensemble improvement: +{improvement:.4f} "
                  f"(+{improvement/best_single_r2*100:.1f}%)")
        else:
            print(f"\n  ⚠️ Single model outperformed ensemble by {-improvement:.4f}")
    
    # Overall best
    best_model = max(results, key=lambda k: results[k]['r2'])
    print(f"\n  🏆 Overall best: {best_model} (R² = {results[best_model]['r2']:.4f})")


def print_results_table(results):
    """
    Print formatted results table.
    
    Args:
        results: Dictionary mapping model names to metric dicts
    """
    print(f"\n{'Model':<25} {'R²':>8} {'MAE (h)':>10} {'MAE (d)':>10} "
          f"{'RMSE (h)':>10} {'RMSE (d)':>10}")
    print("-" * 78)
    
    for name, metrics in results.items():
        mae_d = metrics['mae'] / 24
        rmse_d = metrics['rmse'] / 24
        print(f"{name:<25} {metrics['r2']:>8.4f} {metrics['mae']:>10.1f} "
              f"{mae_d:>10.2f} {metrics['rmse']:>10.1f} {rmse_d:>10.2f}")
    
    print("-" * 78)
