"""
main.py
=======
Main pipeline for Hospital Length of Stay Prediction.

Pipeline:
1. Spark preprocessing (load, clean, feature engineering)
2. Train Gradient Boosting (sklearn)
3. Train Neural Network (PyTorch)
4. Ensemble methods (simple, weighted, stacking)
5. Evaluate and report results

Usage:
    python main.py
    
    # Or with spark-submit on cluster:
    spark-submit main.py
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import Config
from preprocessing import SparkPreprocessor
from model import MSNet1D
from train import (
    train_gradient_boosting,
    train_neural_network,
    evaluate_model,
    predict_neural_network
)
from ensemble import (
    EnsemblePredictor,
    analyze_ensemble,
    print_results_table
)


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    """Execute the complete LOS prediction pipeline."""
    
    print("=" * 80)
    print("HOSPITAL LENGTH OF STAY PREDICTION")
    print("Spark Preprocessing + PyTorch/sklearn Modeling")
    print("=" * 80)
    
    # Configuration
    config = Config()
    config.print_config()
    set_seeds(config.SEED)
    
    device = config.DEVICE
    
    # ================================================================
    # STEP 1: SPARK PREPROCESSING
    # ================================================================
    
    preprocessor = SparkPreprocessor(config)
    
    try:
        data = preprocessor.run()
    finally:
        preprocessor.stop()
    
    # Extract data
    train_pdf = data['train']
    val_pdf = data['val']
    test_pdf = data['test']
    feature_cols = data['feature_cols']
    
    print(f"\n✓ Features: {len(feature_cols)}")
    
    # Prepare numpy arrays
    X_train = train_pdf[feature_cols].values
    y_train = train_pdf['los_hours'].values
    y_train_class = train_pdf['los_class'].values
    
    X_val = val_pdf[feature_cols].values
    y_val = val_pdf['los_hours'].values
    y_val_class = val_pdf['los_class'].values
    
    X_test = test_pdf[feature_cols].values
    y_test = test_pdf['los_hours'].values
    y_test_class = test_pdf['los_class'].values
    
    # ================================================================
    # STEP 2: FEATURE SCALING
    # ================================================================
    
    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  ✓ StandardScaler fitted on {len(X_train):,} samples")
    
    # ================================================================
    # STEP 3: TRAIN GRADIENT BOOSTING
    # ================================================================
    
    gb_model = train_gradient_boosting(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        config
    )
    
    # ================================================================
    # STEP 4: TRAIN NEURAL NETWORKS (Both architectures)
    # ================================================================
    
    # Train Linear version (original)
    print("\n" + "=" * 60)
    print("TRAINING MSNet1D (Linear Version)")
    print("=" * 60)
    nn_model, history = train_neural_network(
        X_train_scaled, y_train, y_train_class,
        X_val_scaled, y_val, y_val_class,
        config, device,
        use_conv=False
    )
    
    # Train Conv+MaxPool version
    print("\n" + "=" * 60)
    print("TRAINING MSNet1D_Conv (MaxPool Version)")
    print("=" * 60)
    nn_model_conv, history_conv = train_neural_network(
        X_train_scaled, y_train, y_train_class,
        X_val_scaled, y_val, y_val_class,
        config, device,
        use_conv=True
    )
    
    # ================================================================
    # STEP 5: EVALUATE BASE MODELS
    # ================================================================
    
    print("\n" + "=" * 60)
    print("EVALUATING BASE MODELS")
    print("=" * 60)
    
    results = {}
    
    # Gradient Boosting
    pred_gb = gb_model.predict(X_test_scaled)
    results['Gradient Boosting'] = evaluate_model(y_test, pred_gb, 'Gradient Boosting')
    print(f"\n  Gradient Boosting - R²: {results['Gradient Boosting']['r2']:.4f}")
    
    # Neural Network (Linear)
    pred_nn = predict_neural_network(nn_model, X_test_scaled, device)
    results['MSNet1D (Linear)'] = evaluate_model(y_test, pred_nn, 'MSNet1D (Linear)')
    print(f"  MSNet1D (Linear) - R²: {results['MSNet1D (Linear)']['r2']:.4f}")
    
    # Neural Network (Conv+MaxPool)
    pred_nn_conv = predict_neural_network(nn_model_conv, X_test_scaled, device)
    results['MSNet1D (MaxPool)'] = evaluate_model(y_test, pred_nn_conv, 'MSNet1D (MaxPool)')
    print(f"  MSNet1D (MaxPool) - R²: {results['MSNet1D (MaxPool)']['r2']:.4f}")
    
    # ================================================================
    # STEP 6: ENSEMBLE METHODS
    # ================================================================
    
    print("\n" + "=" * 60)
    print("ENSEMBLE METHODS")
    print("=" * 60)
    
    # Determine which NN performed better
    linear_r2 = results['MSNet1D (Linear)']['r2']
    conv_r2 = results['MSNet1D (MaxPool)']['r2']
    
    if conv_r2 > linear_r2:
        best_nn = nn_model_conv
        best_nn_name = "MaxPool"
        best_nn_pred = pred_nn_conv
        print(f"\n  Using MSNet1D (MaxPool) for ensemble (R²={conv_r2:.4f} > {linear_r2:.4f})")
    else:
        best_nn = nn_model
        best_nn_name = "Linear"
        best_nn_pred = pred_nn
        print(f"\n  Using MSNet1D (Linear) for ensemble (R²={linear_r2:.4f} >= {conv_r2:.4f})")
    
    ensemble = EnsemblePredictor(gb_model, best_nn, scaler, device)
    
    # Method 1: Simple Average
    print("\n📊 Method 1: Simple Average")
    pred_simple = ensemble.simple_average(X_test)
    results['Ensemble (Simple)'] = evaluate_model(y_test, pred_simple)
    print(f"  R²: {results['Ensemble (Simple)']['r2']:.4f}")
    
    # Method 2: Weighted Average
    print("\n📊 Method 2: Weighted Average")
    weights, val_r2 = ensemble.fit_weights(X_val, y_val)
    pred_weighted = ensemble.weighted_average(X_test)
    results['Ensemble (Weighted)'] = evaluate_model(y_test, pred_weighted)
    print(f"  Test R²: {results['Ensemble (Weighted)']['r2']:.4f}")
    
    # Method 3: Stacking
    print("\n📊 Method 3: Stacking")
    stacking_r2 = ensemble.fit_stacking(X_val, y_val)
    pred_stacking = ensemble.stacking_predict(X_test)
    results['Ensemble (Stacking)'] = evaluate_model(y_test, pred_stacking)
    print(f"  Test R²: {results['Ensemble (Stacking)']['r2']:.4f}")
    
    # ================================================================
    # STEP 7: FINAL RESULTS
    # ================================================================
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print_results_table(results)
    analyze_ensemble(results, y_test)
    
    # Architecture comparison
    print("\n" + "=" * 60)
    print("NEURAL NETWORK ARCHITECTURE COMPARISON")
    print("=" * 60)
    print(f"\n  {'Architecture':<25} {'R²':>10} {'MAE (d)':>10} {'RMSE (d)':>10}")
    print("  " + "-" * 55)
    for name in ['MSNet1D (Linear)', 'MSNet1D (MaxPool)']:
        r = results[name]
        print(f"  {name:<25} {r['r2']:>10.4f} {r['mae']/24:>10.2f} {r['rmse']/24:>10.2f}")
    
    diff = conv_r2 - linear_r2
    if diff > 0:
        print(f"\n  ✓ MaxPool version improved R² by {diff:.4f} ({diff/linear_r2*100:.1f}%)")
    else:
        print(f"\n  ✓ Linear version performed better (or equal) by {-diff:.4f}")
    
    # ================================================================
    # STEP 8: SAVE ARTIFACTS (optional)
    # ================================================================
    
    # Uncomment to save models
    # save_models(gb_model, nn_model, scaler, ensemble, config)
    
    return {
        'results': results,
        'models': {
            'gb': gb_model,
            'nn': nn_model,
            'scaler': scaler,
            'ensemble': ensemble
        },
        'data': {
            'feature_cols': feature_cols,
            'quartiles': data['quartiles']
        },
        'predictions': {
            'test': {
                'Gradient Boosting': pred_gb,
                'MSNet1D': pred_nn,
                'Ensemble (Simple)': pred_simple,
                'Ensemble (Weighted)': pred_weighted,
                'Ensemble (Stacking)': pred_stacking
            },
            'y_test': y_test,
            'X_test': X_test
        },
        'train_val': {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled
        }
    }


def run_failure_analysis(output, gb_model, nn_model, scaler, device, output_dir='failure_analysis'):
    """Run comprehensive failure analysis."""
    from failure_analysis import FailureAnalyzer
    
    # Get data
    y_test = output['predictions']['y_test']
    X_test = output['predictions']['X_test']
    predictions = output['predictions']['test']
    feature_cols = output['data']['feature_cols']
    quartiles = output['data']['quartiles']
    
    # Get train/val data
    y_train = output['train_val']['y_train']
    y_val = output['train_val']['y_val']
    X_train_scaled = output['train_val']['X_train_scaled']
    X_val_scaled = output['train_val']['X_val_scaled']
    
    # Get train/val predictions for overfitting analysis
    pred_train = {
        'Gradient Boosting': gb_model.predict(X_train_scaled),
        'MSNet1D': predict_neural_network(nn_model, X_train_scaled, device)
    }
    pred_val = {
        'Gradient Boosting': gb_model.predict(X_val_scaled),
        'MSNet1D': predict_neural_network(nn_model, X_val_scaled, device)
    }
    
    # Create analyzer
    analyzer = FailureAnalyzer(
        y_true=y_test,
        predictions=predictions,
        X_test=X_test,
        feature_names=feature_cols,
        y_train=y_train,
        pred_train=pred_train,
        y_val=y_val,
        pred_val=pred_val,
        quartiles=quartiles
    )
    
    # Run full analysis
    analyzer.run_full_analysis(output_dir=output_dir)
    
    return analyzer


def save_models(gb_model, nn_model, scaler, ensemble, config):
    """Save trained models and artifacts."""
    import pickle
    import os
    
    os.makedirs('models', exist_ok=True)
    
    # Save Gradient Boosting
    with open('models/gb_model.pkl', 'wb') as f:
        pickle.dump(gb_model, f)
    print("  ✓ Saved: models/gb_model.pkl")
    
    # Save Neural Network
    torch.save(nn_model.state_dict(), 'models/nn_model.pt')
    print("  ✓ Saved: models/nn_model.pt")
    
    # Save Scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  ✓ Saved: models/scaler.pkl")
    
    # Save Ensemble weights
    import json
    ensemble_config = {
        'weights': ensemble.weights,
        'meta_coef': ensemble.meta_learner.coef_.tolist() if ensemble.meta_learner else None,
        'meta_intercept': ensemble.meta_learner.intercept_ if ensemble.meta_learner else None
    }
    with open('models/ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    print("  ✓ Saved: models/ensemble_config.json")


if __name__ == "__main__":
    output = main()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    # Run failure analysis
    run_analysis = input("\nRun failure analysis? (y/n): ").strip().lower()
    if run_analysis == 'y':
        from config import Config
        config = Config()
        analyzer = run_failure_analysis(
            output, 
            output['models']['gb'],
            output['models']['nn'],
            output['models']['scaler'],
            config.DEVICE,
            output_dir='failure_analysis'
        )
        print("\n✓ Failure analysis complete! Check 'failure_analysis/' directory for plots.")
    print("=" * 80)