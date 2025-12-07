"""
preprocessing.py
================
Spark-based data preprocessing for LOS prediction.

Handles:
- Data loading (clinical + vitals)
- Cleaning and filtering
- Feature engineering
- Target encoding
- Train/val/test splitting
- Conversion to Pandas for modeling
"""

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, avg, stddev, count, log1p, pow as spark_pow,
    regexp_replace, coalesce, broadcast, isnan
)
from pyspark.sql.types import DoubleType


class SparkPreprocessor:
    """
    Spark-based data preprocessing pipeline.
    
    Usage:
        preprocessor = SparkPreprocessor(config)
        data = preprocessor.run()
        preprocessor.stop()
    
    Returns dict with:
        - train, val, test: Pandas DataFrames
        - feature_cols: List of feature column names
        - quartiles: LOS quartile boundaries
        - target_encodings: Stored encodings for inference
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with data paths and parameters
        """
        self.config = config
        self.spark = None
        self.quartiles = None
        self.target_encodings = {}
        
    def create_spark_session(self):
        """Initialize Spark session."""
        self.spark = SparkSession.builder \
            .appName("VitalDB_LOS_Preprocessing") \
            .config("spark.executor.memory", "64g") \
            .config("spark.driver.memory", "32g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        print(f"✓ Spark session created (version {self.spark.version})")
        return self.spark
    
    def load_data(self):
        """Load and join clinical and vitals data."""
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        # Load clinical data
        df_clinical = self.spark.read.csv(
            self.config.CLINICAL_PATH,
            header=True,
            inferSchema=True
        )
        print(f"  Clinical: {df_clinical.count():,} rows")
        
        # Load vitals data
        df_vitals = self.spark.read.csv(
            self.config.VITALS_PATH,
            header=True,
            inferSchema=True
        )
        
        # Standardize join column
        if 'case_id' in df_vitals.columns:
            df_vitals = df_vitals.withColumnRenamed('case_id', 'caseid')
        print(f"  Vitals: {df_vitals.count():,} rows")
        
        # Join datasets
        df = df_clinical.join(df_vitals, on='caseid', how='inner')
        print(f"  Combined: {df.count():,} rows")
        
        return df
    
    def clean_data(self, df):
        """
        Clean data: convert types, handle inequalities, apply exclusions.
        
        Exclusion criteria:
        - LOS >= 1 hour
        - LOS <= 30 days
        - Operation duration >= 0.5 hours
        - Operation duration <= 24 hours
        """
        print("\n" + "=" * 60)
        print("CLEANING DATA")
        print("=" * 60)
        
        # Convert timestamp columns to numeric
        timestamp_cols = ['dis', 'adm', 'casestart', 'caseend', 
                         'anestart', 'aneend', 'opstart', 'opend']
        for col_name in timestamp_cols:
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
        
        # Create target and duration features
        df = df.withColumn('los_hours', (col('dis') - col('opend')) / 3600.0)
        df = df.withColumn('los_days', col('los_hours') / 24.0)
        df = df.withColumn('operation_duration_hours', 
                          (col('opend') - col('opstart')) / 3600.0)
        df = df.withColumn('anesthesia_duration_hours', 
                          (col('aneend') - col('anestart')) / 3600.0)
        df = df.withColumn('preop_time_hours', 
                          (col('opstart') - col('adm')) / 3600.0)
        
        # Explicitly clean known columns that may have inequality signs (">89", "<50", etc.)
        # These columns often have values like '>89' for age or '<50' for lab values
        known_numeric_cols = ['age', 'bmi', 'height', 'weight', 'preop_hb', 'preop_plt', 
                              'preop_pt', 'preop_aptt', 'preop_na', 'preop_k', 'preop_gluc',
                              'preop_alb', 'preop_ast', 'preop_alt', 'preop_bun', 'preop_cr']
        
        for col_name in known_numeric_cols:
            if col_name in df.columns:
                # Cast to string first, then clean, then cast to double
                df = df.withColumn(
                    col_name,
                    regexp_replace(col(col_name).cast('string'), r'^[><]=?', '').cast(DoubleType())
                )
        
        # Clean any remaining string columns with inequality signs
        for field in df.schema.fields:
            col_name = field.name
            if col_name in ['caseid', 'opname', 'dx'] + known_numeric_cols:
                continue
            if str(field.dataType) == 'StringType':
                df = df.withColumn(
                    col_name,
                    regexp_replace(col(col_name), r'^[><]=?', '').cast(DoubleType())
                )
        
        count_before = df.count()
        
        # Apply exclusion criteria
        df = df.filter(
            (col('los_hours') >= 1) &
            (col('los_days') <= 30) &
            (col('operation_duration_hours') >= 0.5) &
            (col('operation_duration_hours') <= 24) &
            (col('los_hours').isNotNull())
        )
        
        count_after = df.count()
        print(f"  Rows: {count_before:,} → {count_after:,}")
        print(f"  Excluded: {count_before - count_after:,} ({(count_before - count_after)/count_before*100:.1f}%)")
        
        return df
    
    def _create_target_encoding(self, df, col_name, target_col='los_hours'):
        """
        Create smoothed target encoding for a categorical column.
        
        Formula: (n * category_mean + m * global_mean) / (n + m)
        """
        smoothing = self.config.TARGET_ENCODING_SMOOTHING
        
        # Global mean
        global_mean = df.select(avg(col(target_col))).first()[0]
        
        # Category statistics
        stats = df.groupBy(col_name).agg(
            avg(col(target_col)).alias('_mean'),
            stddev(col(target_col)).alias('_std'),
            count('*').alias('_count')
        )
        
        # Smoothed encoding
        stats = stats.withColumn(
            f'{col_name}_target_enc',
            (col('_count') * col('_mean') + lit(smoothing) * lit(global_mean)) /
            (col('_count') + lit(smoothing))
        )
        stats = stats.withColumn(f'{col_name}_los_mean', col('_mean'))
        stats = stats.withColumn(f'{col_name}_los_std', 
                                coalesce(col('_std'), lit(0.0)))
        
        # Select join columns
        encoding_df = stats.select(
            col_name,
            f'{col_name}_target_enc',
            f'{col_name}_los_mean',
            f'{col_name}_los_std'
        )
        
        # Store for later use
        self.target_encodings[col_name] = encoding_df
        
        # Broadcast join
        df = df.join(broadcast(encoding_df), on=col_name, how='left')
        
        return df
    
    def engineer_features(self, df):
        """
        Create engineered features.
        
        Creates:
        - Target encodings for categoricals
        - Numeric encodings (ASA, sex, emergency)
        - Interaction features
        - Nonlinear transformations
        - Classification labels (quartiles)
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)
        
        # ========== TARGET ENCODING ==========
        print("\n📊 Target Encoding...")
        categorical_cols = ['department', 'optype', 'asa', 'approach', 
                           'position', 'ane_type']
        
        for cat_col in categorical_cols:
            if cat_col in df.columns:
                df = self._create_target_encoding(df, cat_col)
                n_unique = df.select(cat_col).distinct().count()
                print(f"  ✓ {cat_col}: {n_unique} categories")
        
        # ========== NUMERIC ENCODINGS ==========
        print("\n📊 Numeric Encodings...")
        
        if 'asa' in df.columns:
            df = df.withColumn(
                'asa_numeric',
                regexp_replace(col('asa').cast('string'), r'[^0-9]', '').cast(DoubleType())
            )
            print("  ✓ asa_numeric")
        
        if 'sex' in df.columns:
            df = df.withColumn(
                'sex_male',
                when(col('sex').cast('string').isin(['M', 'Male', 'male', '1']), 1.0).otherwise(0.0)
            )
            print("  ✓ sex_male")
        
        if 'emop' in df.columns:
            # Cast to string first, then compare with string values only
            df = df.withColumn(
                'emergency',
                when(col('emop').cast('string').isin(['1', 'Y', 'Yes', 'yes']), 1.0).otherwise(0.0)
            )
            print("  ✓ emergency")
        
        # ========== INTERACTION FEATURES ==========
        print("\n📊 Interaction Features...")
        
        if 'age' in df.columns and 'asa_numeric' in df.columns:
            df = df.withColumn('age_asa', col('age') * col('asa_numeric'))
            print("  ✓ age × ASA")
        
        if 'operation_duration_hours' in df.columns and 'asa_numeric' in df.columns:
            df = df.withColumn('opdur_asa', 
                              col('operation_duration_hours') * col('asa_numeric'))
            print("  ✓ opdur × ASA")
        
        if 'age' in df.columns and 'operation_duration_hours' in df.columns:
            df = df.withColumn('age_opdur', 
                              col('age') * col('operation_duration_hours'))
            print("  ✓ age × opdur")
        
        # ========== NONLINEAR TRANSFORMATIONS ==========
        print("\n📊 Nonlinear Transformations...")
        
        if 'bmi' in df.columns:
            df = df.withColumn(
                'bmi_extreme',
                when((col('bmi') < 18.5) | (col('bmi') > 35), 1.0).otherwise(0.0)
            )
            df = df.withColumn('bmi_squared', spark_pow(col('bmi'), 2))
            print("  ✓ bmi_extreme, bmi_squared")
        
        if 'operation_duration_hours' in df.columns:
            df = df.withColumn('opdur_log', log1p(col('operation_duration_hours')))
            df = df.withColumn('opdur_squared', 
                              spark_pow(col('operation_duration_hours'), 2))
            print("  ✓ opdur_log, opdur_squared")
        
        if 'age' in df.columns:
            df = df.withColumn('age_squared', spark_pow(col('age'), 2))
            df = df.withColumn('age_log', log1p(col('age')))
            print("  ✓ age_squared, age_log")
        
        # ========== CLASSIFICATION LABELS ==========
        print("\n📊 Classification Labels...")
        
        self.quartiles = df.approxQuantile('los_hours', [0.25, 0.5, 0.75], 0.01)
        print(f"  Quartiles: {[f'{q:.1f}h' for q in self.quartiles]}")
        
        df = df.withColumn(
            'los_class',
            when(col('los_hours') <= self.quartiles[0], 0)
            .when(col('los_hours') <= self.quartiles[1], 1)
            .when(col('los_hours') <= self.quartiles[2], 2)
            .otherwise(3)
            .cast(DoubleType())
        )
        
        return df
    
    def select_features(self, df):
        """
        Select numeric features for modeling.
        
        Excludes ID columns, targets, raw categoricals, and timestamps.
        """
        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)
        
        exclude_patterns = [
            'caseid', 'subject', 'los_hours', 'los_days', 'los_class',
            'dis', 'adm', 'opstart', 'opend', 'anestart', 'aneend',
            'casestart', 'caseend', 'opname', 'dx'
        ]
        
        categorical_raw = ['department', 'optype', 'asa', 'approach',
                          'position', 'ane_type', 'sex', 'emop']
        
        feature_cols = []
        total_rows = df.count()
        
        for field in df.schema.fields:
            col_name = field.name
            
            if any(pat in col_name.lower() for pat in exclude_patterns):
                continue
            if col_name in categorical_raw:
                continue
            
            dtype_str = str(field.dataType)
            if dtype_str in ['DoubleType()', 'IntegerType()', 'FloatType()', 'LongType()']:
                # Only use isNull for safety - isnan can cause issues in Spark 4.0
                null_count = df.filter(col(col_name).isNull()).count()
                if null_count / total_rows < 0.5:
                    feature_cols.append(col_name)
        
        print(f"  Selected: {len(feature_cols)} features")
        
        return feature_cols
    
    def split_data(self, df):
        """Split into train/val/test sets."""
        print("\n" + "=" * 60)
        print("DATA SPLITTING")
        print("=" * 60)
        
        df_train, df_val, df_test = df.randomSplit(
            [self.config.TRAIN_RATIO, self.config.VAL_RATIO, self.config.TEST_RATIO],
            seed=self.config.SEED
        )
        
        print(f"  Train: {df_train.count():,}")
        print(f"  Val:   {df_val.count():,}")
        print(f"  Test:  {df_test.count():,}")
        
        return df_train, df_val, df_test
    
    def to_pandas(self, df, feature_cols):
        """Convert Spark DataFrame to Pandas."""
        columns = feature_cols + ['los_hours', 'los_class']
        pdf = df.select(columns).toPandas()
        
        # Handle remaining nulls
        for col_name in feature_cols:
            if pdf[col_name].isna().any():
                pdf[col_name] = pdf[col_name].fillna(pdf[col_name].median())
        
        # Handle infinities
        pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()
        
        return pdf
    
    def run(self):
        """
        Execute complete preprocessing pipeline.
        
        Returns:
            dict with train/val/test DataFrames and metadata
        """
        self.create_spark_session()
        
        df = self.load_data()
        df = self.clean_data(df)
        df = self.engineer_features(df)
        feature_cols = self.select_features(df)
        
        df = df.cache()
        
        df_train, df_val, df_test = self.split_data(df)
        
        print("\n" + "=" * 60)
        print("CONVERTING TO PANDAS")
        print("=" * 60)
        
        train_pdf = self.to_pandas(df_train, feature_cols)
        val_pdf = self.to_pandas(df_val, feature_cols)
        test_pdf = self.to_pandas(df_test, feature_cols)
        
        print(f"  Train: {len(train_pdf):,} rows")
        print(f"  Val:   {len(val_pdf):,} rows")
        print(f"  Test:  {len(test_pdf):,} rows")
        
        df.unpersist()
        
        return {
            'train': train_pdf,
            'val': val_pdf,
            'test': test_pdf,
            'feature_cols': feature_cols,
            'quartiles': self.quartiles,
            'target_encodings': self.target_encodings
        }
    
    def stop(self):
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()
            print("\n✓ Spark session stopped")