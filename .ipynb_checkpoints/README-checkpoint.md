# Hospital Length of Stay Prediction with MSNet1D

## Overview

This project predicts hospital length of stay (LOS) following surgical operations using the VitalDB dataset. We developed MSNet1D, a multi-scale neural network architecture, and conducted systematic ablation studies comparing neural networks to gradient boosting methods.

### Key Findings

1. **Neural networks outperform Gradient Boosting on MAE**: All neural network variants achieved lower mean absolute error (2.04–2.14 days) than Gradient Boosting (2.32 days), representing a 9–13% reduction in prediction error despite lower R² scores.

2. **ASPP hurts performance on aggregated features**: Removing the ASPP module improved R² from 0.4223 to 0.4443. ASPP was designed for spatial/temporal structure that doesn't exist in mean-aggregated vital signs.

3. **Multi-Scale Blocks are essential**: Removing them degraded performance below a simple MLP baseline (R² 0.4046 vs 0.4182).

4. **Huber loss is critical**: MSE-based losses failed dramatically on right-skewed LOS distributions.

## Results Summary

| Model | R² | MAE (days) | RMSE (days) | Parameters |
|-------|-----|------------|-------------|------------|
| Gradient Boosting | **0.4572** | 2.32 | **3.69** | --- |
| MSNet1D (No ASPP) | 0.4443 | 2.08 | 3.73 | 119,957 |
| MSNet1D (Conv + MaxPool) | 0.4260 | 2.14 | 3.79 | 126,037 |
| MSNet1D (Full Model) | 0.4223 | **2.04** | 3.80 | 129,445 |
| Simple MLP | 0.4182 | 2.07 | 3.82 | 26,261 |
| MSNet1D (No MSBlocks) | 0.4046 | 2.11 | 3.86 | 40,037 |

## Dataset

**VitalDB** ([PhysioNet](https://physionet.org/content/vitaldb/1.0.0/))
- 6,388 non-cardiac surgical cases from Seoul National University Hospital (2016-2017)
- 74 clinical variables + 44 vital sign channels from Solar™ 8000M monitor
- Vital signs aggregated to mean values across surgery duration
- LOS range: 1 hour to 30 days (after filtering)
- LOS quartiles: Q1=60.0h, Q2=108.5h, Q3=175.4h

## Project Structure

```


├── Code/
│   ├── config.py                   # Configuration and hyperparameters
│   ├── preprocessing.py            # Spark-based data preprocessing
│   ├── ablation_study.py           # Main ablation study script
│   ├── training_ablation_simple.py # Training config ablation (loss, dropout)

```

## Installation

```bash
# Required packages
pip install torch numpy pandas scikit-learn matplotlib

# For preprocessing (optional - requires Spark)
pip install pyspark
```

## Usage

### Run Architecture Ablation Study

```bash
# Quick run (6 experiments)
python ablation_study.py --quick

# All experiments
python ablation_study.py --all

# Specific experiments
python ablation_study.py --exp no_aspp gradient_boosting simple_mlp

# List available experiments
python ablation_study.py --list
```

### Run Training Configuration Ablation

```bash
python training_ablation_simple.py
```

Outputs:
- `training_curves.png` - Training/validation loss curves
- `training_ablation_results.csv` - Raw metrics

## Configuration

Key hyperparameters (in `Config` class):

```python
# Optimized settings
DROPOUT = 0.3
REG_WEIGHT = 0.4  # 40% regression loss
CLS_WEIGHT = 0.6  # 60% classification loss

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 25

# Loss functions
# Regression: HuberLoss (delta=1.0)
# Classification: FocalLoss (gamma=2.0)
```

## Architecture

### MSNet1D

```
Input (111 features)
    ↓
Input Projection (111 → 128)
    ↓
Multi-Scale Block 1 (128 → 64)
├── Path 1: Direct (1 layer)
├── Path 2: Medium (2 layers)  
└── Path 3: Bottleneck (3 layers)
    ↓
Multi-Scale Block 2 (64 → 64)
├── Path 1: Direct
├── Path 2: Medium
└── Path 3: Bottleneck
    ↓
[ASPP Module - OPTIONAL, hurts performance]
    ↓
Shared Embedding (64 → 32)
    ↓
┌───────────────┬───────────────┐
↓               ↓
Regression Head    Classification Head
(32 → 16 → 1)     (32 → 16 → 4)
↓               ↓
LOS (hours)        LOS Quartile
```

### Why No ASPP?

ASPP (Atrous Spatial Pyramid Pooling) was designed for image segmentation to capture multi-scale spatial context. When vital signs are aggregated to mean values:
- No temporal/spatial structure exists for ASPP to exploit
- The ~9,500 extra parameters lead to overfitting
- R² drops from 0.4443 → 0.4223 when ASPP is added

### Why Multi-Scale Blocks Work

The parallel paths capture relationships at different complexity levels:
- **Path 1 (Direct)**: Simple linear relationships (e.g., age → LOS)
- **Path 2 (Medium)**: Moderate interactions (e.g., ASA × duration)
- **Path 3 (Bottleneck)**: Complex patterns with regularization

## Data Paths

Update these in `config.py` or `ablation_study.py`:

```python
CLINICAL_PATH = '/path/to/clinical_data.csv'
VITALS_ALL_MEAN_PATH = '/path/to/all_mean_features/'
VITALS_LAST_HOUR_PATH = '/path/to/last_hour_mean_features/'
```

## Feature Engineering

### Clinical Features
- Demographics: age, sex, BMI, height, weight
- ASA physical status classification
- Surgical descriptors: department, approach, position, operation type
- Preoperative labs: hemoglobin, platelets, electrolytes, renal/liver function
- Timing: operation duration, anesthesia duration

### Vital Sign Features
- 44 channels from Solar™ 8000M monitor
- Aggregated to mean values across surgery
- Includes: HR, BP (sys/dia/mean), SpO2, RR, EtCO2, temperature

### Engineered Features
- Target encoding for categorical variables (smoothing=10)
- Interactions: age × ASA, duration × ASA, age × duration
- Nonlinear transforms: log(age), age², log(duration), duration²
- BMI extreme indicator (BMI < 18.5 or > 30)

## Experiments

| Experiment | Description |
|------------|-------------|
| `full_model` | MSNet1D with Multi-Scale Blocks + ASPP |
| `no_aspp` | MSNet1D without ASPP (best neural network) |
| `no_msblocks` | ASPP only, MLP backbone |
| `simple_mlp` | Plain MLP baseline |
| `conv_maxpool` | Conv1D + MaxPool variant |
| `gradient_boosting` | Scikit-learn GradientBoostingRegressor |

## Training Ablation Results

### Loss Functions
| Loss | Val Loss | Stability | Gap |
|------|----------|-----------|-----|
| **Huber + Focal** | **1.284** | **0.019** | **0.468** |
| Huber + CE | 1.467 | 0.021 | 0.447 |
| MSE + Focal | 8.829 | 0.135 | 5.238 |
| MSE + CE | 8.885 | 0.122 | 5.302 |

**Conclusion**: Huber loss is essential; MSE fails on right-skewed LOS.

### Dropout Rates
| Dropout | Val Loss | Stability | Gap |
|---------|----------|-----------|-----|
| 0.0 | 1.302 | 0.036 | 0.809 |
| 0.1 | 1.284 | 0.019 | 0.468 |
| 0.2 | 1.265 | 0.011 | 0.275 |
| **0.3** | **1.258** | **0.008** | **0.212** |
| 0.5 | 1.277 | 0.014 | 0.118 |

**Conclusion**: Dropout 0.3 provides best stability and generalization.

## Citation

If you use this code, please cite:

```bibtex
@article{mathai2024los,
  title={Neural Networks Achieve Lower Prediction Error Than Gradient Boosting for Hospital Length of Stay},
  author={Mathai, Madelyn},
  journal={arXiv preprint},
  year={2024}
}
```

## References

- Lee et al. (2022). VitalDB, a high-fidelity multi-parameter vital signs database. Scientific Reports.
- Grinsztajn et al. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS.
- Chen et al. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation (DeepLab/ASPP).
- Lin et al. (2017). Focal Loss for Dense Object Detection.

## License

This project uses the VitalDB dataset, which is available under the PhysioNet Credentialed Health Data License.

## Contact

Madelyn Mathai - euh7ys@virginia.edu