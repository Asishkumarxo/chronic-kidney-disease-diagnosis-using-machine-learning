"""
Standalone Data Preprocessing Module for Chronic Kidney Disease (CKD) Staging System.

This script consolidates all data preprocessing operations into a single, self-contained, 
reusable module. It handles:
1. Raw CSV Data Loading & Column Name Standardization.
2. Target Variable Cleaning & Label Encoding (0 to 4 staging mapping).
3. Categorical Text Field Standardization.
4. Preventative Data Leakage Mitigation (eGFR Feature Exclusion).
5. Missing Value Imputation (Median for numeric features, Mode for categorical features).
6. Feature Transformation via Scikit-Learn Pipeline (StandardScaler for numeric, OneHotEncoder for categorical).
7. Transformed Feature Matrix Assembly & Output Generation.

Note: This file is completely independent and standalone. It does not modify or alter 
any existing scripts or files in the project.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Define standard feature column groupings
NUMERIC_COLS = [
    'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 
    'Serum_Creatinine', 'Blood_Urea_Nitrogen', 'Urine_Albumin', 
    'Urine_Protein', 'Albumin_Creatinine_Ratio', 'Urine_Specific_Gravity', 
    'Sodium', 'Potassium', 'Calcium', 'Phosphorus', 'Chloride', 
    'Bicarbonate', 'Hemoglobin', 'RBC_Count', 'WBC_Count', 
    'Platelet_Count', 'Packed_Cell_Volume', 'Blood_Glucose_Random', 
    'Fasting_Glucose', 'HbA1c', 'Cholesterol', 'Triglycerides', 
    'Serum_Albumin', 'Total_Protein'
]

CATEGORICAL_COLS = [
    'Gender', 'Diabetes', 'Hypertension', 'Smoking_Status', 'Family_History_Kidney'
]

TARGET_MAPPING = {
    'Healthy Kidney': 0,
    'Mild CKD (Stage 1-2)': 1,
    'Moderate CKD (Stage 3)': 2,
    'Severe CKD (Stage 4)': 3,
    'Kidney Failure (Stage 5)': 4
}


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame strings, target encoding, and categorical fields.
    """
    df = df.copy()
    
    # 1. Strip column name whitespace
    df.columns = df.columns.str.strip()
    
    # 2. Fix target Unicode en-dash and map to stage integers
    if 'Target' in df.columns:
        if df['Target'].dtype == object or isinstance(df['Target'].dtype, pd.StringDtype):
            df['Target'] = df['Target'].astype(str).str.replace('\u2013', '-', regex=False).str.strip()
            df['Target'] = df['Target'].map(TARGET_MAPPING)
            
        # Drop rows where target couldn't be mapped
        df = df.dropna(subset=['Target'])
        df['Target'] = df['Target'].astype(int)
        
    # 3. Standardize categorical text fields
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if col == 'Gender':
                def parse_gender(val):
                    s = str(val).strip().lower()
                    if s in ['1', '1.0', 'male', 'm']:
                        return 'Male'
                    elif s in ['0', '0.0', 'female', 'f']:
                        return 'Female'
                    return str(val).strip().capitalize()
                df[col] = df[col].apply(parse_gender)
            else:
                df[col] = df[col].astype(str).str.strip().str.capitalize()
                
    # 4. Standardize any remaining string columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    return df


def Separate_features_and_target(df: pd.DataFrame):
    """
    Separates feature matrix X and target vector y.
    Mitigates Data Leakage by explicitly removing eGFR column.
    """
    y = df['Target'] if 'Target' in df.columns else None
    
    # Exclude Target and eGFR to prevent data leakage
    drop_cols = [col for col in ['Target', 'eGFR'] if col in df.columns]
    X = df.drop(columns=drop_cols)
    
    return X, y


def build_standalone_preprocessor() -> ColumnTransformer:
    """
    Constructs scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    # Numeric transformer pipeline: Median Imputation + Z-score Standardization
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer pipeline: Mode Imputation + One-Hot Encoding
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_COLS),
            ('cat', categorical_pipeline, CATEGORICAL_COLS)
        ]
    )
    
    return preprocessor


def preprocess_dataset(input_csv_path: str):
    """
    Complete end-to-end data preprocessing function for a given dataset CSV.
    """
    print(f"Loading raw dataset from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found at: {input_csv_path}")
        
    raw_df = pd.read_csv(input_csv_path, encoding='utf-8')
    print(f"Raw DataFrame shape: {raw_df.shape}")
    
    # 1. Clean raw text & target
    cleaned_df = clean_raw_dataframe(raw_df)
    
    # 2. Extract X and y (excluding eGFR to prevent leakage)
    X, y = Separate_features_and_target(cleaned_df)
    print(f"Features matrix shape X: {X.shape}, Target vector shape y: {y.shape if y is not None else None}")
    
    # 3. Build & fit standalone preprocessor
    preprocessor = build_standalone_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    all_feature_names = NUMERIC_COLS + encoded_cat_names
    
    # Create clean preprocessed DataFrame
    X_preprocessed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    
    return X_preprocessed_df, y, preprocessor


if __name__ == '__main__':
    print("=== Standalone CKD Data Preprocessing Pipeline ===")
    
    data_dir = r"c:\Users\donip\Desktop\CKD diagnosing using ML\data"
    train_csv = os.path.join(data_dir, "Training_CKD_dataset.csv")
    
    if os.path.exists(train_csv):
        X_proc, y_proc, preprocessor_obj = preprocess_dataset(train_csv)
        print("\nPreprocessing successful!")
        print(f"Transformed Feature Matrix shape: {X_proc.shape}")
        print(f"Number of Numeric Features processed: {len(NUMERIC_COLS)}")
        print(f"Number of Categorical Features encoded: {len(CATEGORICAL_COLS)}")
        print(f"Total Columns after One-Hot Encoding: {X_proc.shape[1]}")
        print("\nSample Preprocessed Features (First 3 rows, top 5 columns):")
        print(X_proc.iloc[:3, :5])
    else:
        print(f"Dataset file not found at {train_csv}. Please verify file location.")
