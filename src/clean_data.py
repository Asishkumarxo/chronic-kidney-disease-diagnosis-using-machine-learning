import os
import pandas as pd

def clean_csv_dataset(input_path, output_path):
    print(f"Cleaning dataset: {input_path} -> {output_path}...")
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return False
        
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # 1. Clean column names (strip potential spaces)
    df.columns = df.columns.str.strip()
    
    # 2. Fix the en-dash encoding in Target labels
    if 'Target' in df.columns:
        # Standardize target values by replacing unicode en-dash with standard hyphen
        df['Target'] = df['Target'].str.replace('\u2013', '-', regex=False).str.strip()
        print("Standardized target values to:")
        print(df['Target'].value_counts())
        
    # 3. Standardize categorical features (Yes/No fields) to standard capitalized form ("Yes" / "No")
    categorical_fields = ['Diabetes', 'Hypertension', 'Smoking_Status', 'Family_History_Kidney']
    for col in categorical_fields:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()
            
    # 4. Standardize text representation of other nominal columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # 5. Handle missing values if any exist (safety check)
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Found {missing_count} missing values. Performing imputation...")
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Cleaned dataset saved successfully. Shape: {df.shape}")
    return True

if __name__ == '__main__':
    data_dir = r"c:\Users\donip\Desktop\CKD diagnosing using ML\data"
    
    # File paths
    train_input = os.path.join(data_dir, "Training_CKD_dataset.csv")
    train_output = os.path.join(data_dir, "Training_CKD_dataset_clean.csv")
    
    test_input = os.path.join(data_dir, "Testing_CKD_dataset.csv")
    test_output = os.path.join(data_dir, "Testing_CKD_dataset_clean.csv")
    
    print("=== Starting Dataset Cleaning ===")
    clean_csv_dataset(train_input, train_output)
    print("\n---------------------------------\n")
    clean_csv_dataset(test_input, test_output)
    print("\n=== Dataset Cleaning Completed ===")
