"""
config.py
=========
Configuration settings for LOS prediction pipeline.
"""

import torch


class Config:
    """
    Configuration for the LOS prediction pipeline.
    
    Sections:
    - Data paths
    - Data split ratios
    - Neural network hyperparameters
    - Feature engineering parameters
    """
    
    # ========== DATA PATHS ==========
    CLINICAL_PATH = '/sfs/ceph/standard/ds7200-apt4c/vdbd_data/physionet_download/files/vitaldb/1.0.0/clinical_data.csv'
    VITALS_PATH = '/sfs/ceph/standard/ds7200-apt4c/vdbd_data/intermediate_data/last_hour_mean_features/'
    VITALS_PARQUET_FOLDER = "/sfs/ceph/standard/ds7200-apt4c/vdbd_data/physionet_download/files/vitaldb/1.0.0/vital_files/parquet/"
    # ========== RANDOM SEED ==========
    SEED = 42
    
    # ========== DATA SPLIT ==========
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ========== NEURAL NETWORK ==========
    BATCH_SIZE = 32
    MAX_EPOCHS = 150
    EARLY_STOPPING_PATIENCE = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3
    
    # ========== FEATURE ENGINEERING ==========
    TARGET_ENCODING_SMOOTHING = 20
    
    # ========== DEVICE ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        """Print configuration settings."""
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"  Seed: {cls.SEED}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Split: {cls.TRAIN_RATIO}/{cls.VAL_RATIO}/{cls.TEST_RATIO}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Max epochs: {cls.MAX_EPOCHS}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Weight decay: {cls.WEIGHT_DECAY}")
        print(f"  Dropout: {cls.DROPOUT}")
        print(f"  Early stopping patience: {cls.EARLY_STOPPING_PATIENCE}")


# For convenience, create a default config instance
config = Config()
