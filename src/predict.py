import os
import sys
import pandas as pd
import joblib

def predict_ckd_stage(patient_data):
    model_path = r"c:\Users\donip\Desktop\CKD diagnosing using ML\models\ckd_stage_model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_model.py first.")
        return None
        
    pipeline = joblib.load(model_path)
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame([patient_data])
    
    # Drop Target and eGFR if they are mistakenly passed
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])
    if 'eGFR' in df.columns:
        df = df.drop(columns=['eGFR'])
        
    # Standardize types and whitespace just in case
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == 'object':
            df[col] = df[col].astype(str).str.strip()
            
    # Predict
    pred_class = pipeline.predict(df)[0]
    probabilities = pipeline.predict_proba(df)[0]
    
    target_names = [
        'Healthy Kidney', 
        'Mild CKD (Stage 1-2)', 
        'Moderate CKD (Stage 3)', 
        'Severe CKD (Stage 4)', 
        'Kidney Failure (Stage 5)'
    ]
    
    label = target_names[pred_class]
    
    # Overall risk of CKD = 1.0 - probability of class 0 (Healthy Kidney)
    ckd_risk = 1.0 - probabilities[0]
    
    return {
        'prediction': int(pred_class),
        'label': label,
        'probability': float(ckd_risk),
        'stage_probabilities': {target_names[i]: float(probabilities[i]) for i in range(len(target_names))}
    }

if __name__ == '__main__':
    # Test case with mock healthy patient parameters
    healthy_patient = {
        'Age': 40, 'Gender': 1, 'BMI': 23, 'Systolic_BP': 110, 'Diastolic_BP': 75, 'Heart_Rate': 72,
        'Serum_Creatinine': 0.8, 'Blood_Urea_Nitrogen': 12, 'Urine_Albumin': 5, 'Urine_Protein': 1,
        'Albumin_Creatinine_Ratio': 12, 'Urine_Specific_Gravity': 1.025, 'Sodium': 140, 'Potassium': 4.1,
        'Calcium': 9.6, 'Phosphorus': 3.4, 'Chloride': 101, 'Bicarbonate': 24, 'Hemoglobin': 15.2,
        'RBC_Count': 5.1, 'WBC_Count': 6500, 'Platelet_Count': 240000, 'Packed_Cell_Volume': 44,
        'Blood_Glucose_Random': 95, 'Fasting_Glucose': 88, 'HbA1c': 5.2, 'Cholesterol': 175,
        'Triglycerides': 115, 'Serum_Albumin': 4.3, 'Total_Protein': 7.1, 'Diabetes': 'No',
        'Hypertension': 'No', 'Smoking_Status': 'No', 'Family_History_Kidney': 'No'
    }
    
    print("Testing Mock Patient Preset:")
    res = predict_ckd_stage(healthy_patient)
    if res:
        print(f"Prediction class: {res['prediction']}")
        print(f"Staging Diagnosis: {res['label']}")
        print(f"CKD Risk: {res['probability']*100:.2f}%")
        print("Probabilities:")
        for k, v in res['stage_probabilities'].items():
            print(f"  {k}: {v*100:.2f}%")
