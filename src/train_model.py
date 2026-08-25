import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, encoding='utf-8')
    
    target_mapping = {
        'Healthy Kidney': 0,
        'Mild CKD (Stage 1-2)': 1,
        'Moderate CKD (Stage 3)': 2,
        'Severe CKD (Stage 4)': 3,
        'Kidney Failure (Stage 5)': 4
    }
    
    df['Target'] = df['Target'].map(target_mapping)
    if df['Target'].isnull().any():
        print("Warning: Found unmapped targets, dropping rows.")
        df = df.dropna(subset=['Target'])
        
    df['Target'] = df['Target'].astype(int)
    
    # Exclude eGFR to make prediction rely on actual biomarkers (prevent leakage/cheating)
    X = df.drop(columns=['Target', 'eGFR'])
    y = df['Target']
    
    return X, y

def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def train_and_evaluate():
    data_dir = r"c:\Users\donip\Desktop\CKD diagnosing using ML\data"
    train_path = os.path.join(data_dir, "Training_CKD_dataset_clean.csv")
    test_path = os.path.join(data_dir, "Testing_CKD_dataset_clean.csv")
    
    X_train, y_train = load_and_preprocess_data(train_path)
    X_test, y_test = load_and_preprocess_data(test_path)
    
    # Define columns
    numeric_cols = [
        'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 
        'Serum_Creatinine', 'Blood_Urea_Nitrogen', 'Urine_Albumin', 
        'Urine_Protein', 'Albumin_Creatinine_Ratio', 'Urine_Specific_Gravity', 
        'Sodium', 'Potassium', 'Calcium', 'Phosphorus', 'Chloride', 
        'Bicarbonate', 'Hemoglobin', 'RBC_Count', 'WBC_Count', 
        'Platelet_Count', 'Packed_Cell_Volume', 'Blood_Glucose_Random', 
        'Fasting_Glucose', 'HbA1c', 'Cholesterol', 'Triglycerides', 
        'Serum_Albumin', 'Total_Protein'
    ]
    
    categorical_cols = ['Gender', 'Diabetes', 'Hypertension', 'Smoking_Status', 'Family_History_Kidney']
    
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # Use multi-class classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
    }
    
    best_model_name = None
    best_model = None
    best_score = -1
    
    print("\nEvaluating models on training set (5-Fold Cross Validation):")
    for name, clf in classifiers.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = np.mean(scores)
        print(f"{name}: Mean CV Accuracy = {mean_score:.4f} (std: {np.std(scores):.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_model = pipeline
            
    print(f"\nBest Model Selected: {best_model_name} with CV Accuracy: {best_score:.4f}")
    
    # Train the best model on full training set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    target_names = [
        'Healthy Kidney', 
        'Mild CKD (Stage 1-2)', 
        'Moderate CKD (Stage 3)', 
        'Severe CKD (Stage 4)', 
        'Kidney Failure (Stage 5)'
    ]
    
    print(f"\nEvaluation on Independent Test Set ({best_model_name}):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC (OVR Weighted): {roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the model
    models_dir = r"c:\Users\donip\Desktop\CKD diagnosing using ML\models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "ckd_stage_model.joblib")
    joblib.dump(best_model, model_save_path)
    print(f"\nSaved trained multi-class pipeline to: {model_save_path}")
    
    # Print feature importance for Random Forest
    if best_model_name == 'Random Forest':
        cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        encoded_cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        all_features = numeric_cols + encoded_cat_features
        importances = best_model.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print("\nTop 10 Feature Importances:")
        print(feature_importance_df.head(10).to_string(index=False))

if __name__ == '__main__':
    train_and_evaluate()
