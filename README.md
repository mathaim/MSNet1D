# Hospital Length of Stay Prediction with MSNet1D

## Overview

This project predicts hospital length of stay (LOS) following surgical operations using the VitalDB dataset. We developed MSNet1D, a multi-scale neural network architecture, and conducted systematic ablation studies comparing neural networks to gradient boosting methods.

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
- Stone et al. (2022). A systematic review of the prediction of hospital length of stay: Towards a unified framework. PLOS Digital Health.
- Shwartz-Ziv & Armon (2022). Tabular data: Deep learning is not all you need. Information Fusion.
- Grinsztajn et al. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS.
- Chen et al. (2023). A deep learning approach for inpatient length of stay and mortality prediction. Journal of Biomedical Informatics.
- Rocheteau et al. (2021). Temporal pointwise convolutional networks for length of stay prediction in the intensive care unit. CHIL Conference.
- He et al. (2014). Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition. ECCV.
- Chen et al. (2018). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. IEEE TPAMI.
- Lee et al. (2022). VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Scientific Data.
- Lin et al. (2017). Focal loss for dense object detection. ICCV.
- Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation.
- Mei et al. (2025). Prolonged length of stay and its predictors among surgical patients. BMC Surgery.
- Belayneh et al. (2023). Prolonged hospital stay and associated factors among surgical patients. Patient Safety in Surgery.
- Stewart et al. (2021). The impact of prolonged length of stay on patient outcomes. Journal of Surgical Research.
- Anthropic (2024). Claude. Large language model. Available at: https://www.anthropic.com/claude

## License

This project uses the VitalDB dataset, which is available under the PhysioNet Credentialed Health Data License.

## Contact

Madelyn Mathai - euh7ys@virginia.edu