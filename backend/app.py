import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)  # Enable Cross-Origin Resource Sharing for all domains

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Load the trained multi-class model pipeline
model_path = r"c:\Users\donip\Desktop\CKD diagnosing using ML\models\ckd_stage_model.joblib"
pipeline = None

if os.path.exists(model_path):
    try:
        pipeline = joblib.load(model_path)
        print(f"Multi-class stage model successfully loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model not found at {model_path}. Please run train_model.py first.")

# Define the expected feature lists (matching clean CSV columns)
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

CATEGORICAL_COLS = ['Gender', 'Diabetes', 'Hypertension', 'Smoking_Status', 'Family_History_Kidney']
ALL_COLS = NUMERIC_COLS + CATEGORICAL_COLS

def calculate_ckd_epi(age, gender, creatinine):
    """
    Calculates the 2021 CKD-EPI Creatinine Equation (race-free clinical standard).
    gender: 1 for Male, 0 for Female.
    creatinine: mg/dL.
    """
    if creatinine <= 0:
        creatinine = 0.6  # clinical safety minimum
        
    if gender == 1:  # Male
        k = 0.9
        alpha = -0.302
        gender_mult = 1.0
    else:  # Female
        k = 0.7
        alpha = -0.241
        gender_mult = 1.012
        
    # Formula components
    cr_term = min(creatinine / k, 1) ** alpha
    cr_term_high = max(creatinine / k, 1) ** -1.200
    age_term = 0.9938 ** age
    
    gfr = 142 * cr_term * cr_term_high * age_term * gender_mult
    return gfr

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': pipeline is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    global pipeline
    if pipeline is None:
        if os.path.exists(model_path):
            try:
                pipeline = joblib.load(model_path)
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Model load failed: {str(e)}'}), 500
        else:
            return jsonify({'status': 'error', 'message': 'Model is not available. Please train the model.'}), 500

    try:
        data = request.json
        print("Received prediction request JSON:", data)
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400

        # Construct patient dictionary with proper types
        patient_dict = {}
        
        # Parse numerical inputs
        for col in NUMERIC_COLS:
            val = data.get(col)
            if val is None or val == '' or val == '?':
                patient_dict[col] = np.nan
            else:
                try:
                    patient_dict[col] = float(val)
                except ValueError:
                    patient_dict[col] = np.nan

        # Parse categorical inputs
        for col in CATEGORICAL_COLS:
            val = data.get(col)
            if val is None or val == '' or val == '?':
                # Impute default gender if missing
                if col == 'Gender':
                    patient_dict[col] = 1.0  # default Male
                else:
                    patient_dict[col] = np.nan
            else:
                # Handle special mapping for Gender (convert string/float to float 0 or 1)
                if col == 'Gender':
                    s_val = str(val).strip().lower()
                    if s_val in ['1', '1.0', 'male', 'm']:
                        patient_dict[col] = 1.0
                    elif s_val in ['0', '0.0', 'female', 'f']:
                        patient_dict[col] = 0.0
                    else:
                        patient_dict[col] = 1.0  # fallback
                else:
                    patient_dict[col] = str(val).strip().capitalize() # "Yes" or "No"

        # Convert to DataFrame
        df = pd.DataFrame([patient_dict])
        
        # Ensure exact column ordering matching pipeline
        df = df[ALL_COLS]

        # Get multi-class predictions
        pred = pipeline.predict(df)[0]  # Integer 0-4
        proba = pipeline.predict_proba(df)[0]  # Probabilities of classes 0-4
        
        target_names = [
            'Healthy Kidney', 
            'Mild CKD (Stage 1-2)', 
            'Moderate CKD (Stage 3)', 
            'Severe CKD (Stage 4)', 
            'Kidney Failure (Stage 5)'
        ]
        
        ckd_stage = target_names[pred]
        
        # Overall risk of CKD = sum of probabilities of classes 1 to 4 (which is 1.0 - proba[0])
        prob_ckd = 1.0 - proba[0]
        
        # Risk level categorization
        if pred == 0:
            risk_level = "Low Risk"
        elif pred in [1, 2]:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"
            
        # Clinical eGFR estimation using 2021 CKD-EPI formula
        age_val = float(patient_dict.get('Age')) if not pd.isna(patient_dict.get('Age')) else 50.0
        gender_val = int(patient_dict.get('Gender')) if not pd.isna(patient_dict.get('Gender')) else 1
        sc_val = float(patient_dict.get('Serum_Creatinine')) if not pd.isna(patient_dict.get('Serum_Creatinine')) else 1.0
        
        # Clinical GFR calculation
        gfr = calculate_ckd_epi(age_val, gender_val, sc_val)
        
        # Patient-specific key risk factors computation
        risk_factors = []
        try:
            clf = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            
            importances = None
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
            elif hasattr(clf, 'coef_'):
                importances = np.abs(clf.coef_[0])
                
            if importances is not None:
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                encoded_cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
                
                # Weight Base Features
                base_importances = {}
                for i, col in enumerate(NUMERIC_COLS):
                    base_importances[col] = importances[i]
                
                cat_start_idx = len(NUMERIC_COLS)
                for i, name in enumerate(encoded_cat_features):
                    base_col = None
                    for cat_col in CATEGORICAL_COLS:
                        if name.startswith(cat_col + "_"):
                            base_col = cat_col
                            break
                    if base_col:
                        base_importances[base_col] = base_importances.get(base_col, 0.0) + importances[cat_start_idx + i]
                
                # Normal ranges: lower bound, upper bound, display name
                clinical_ranges = {
                    'Serum_Creatinine': (None, 1.2, 'Serum Creatinine (Elevated)'),
                    'Blood_Urea_Nitrogen': (None, 20.0, 'Blood Urea Nitrogen (Elevated)'),
                    'Urine_Protein': (None, 0.0, 'Urine Protein (Proteinuria)'),
                    'Urine_Albumin': (None, 30.0, 'Urine Albumin (Microalbuminuria)'),
                    'Systolic_BP': (None, 130.0, 'Systolic Blood Pressure (High)'),
                    'Diastolic_BP': (None, 80.0, 'Diastolic Blood Pressure (High)'),
                    'HbA1c': (None, 6.0, 'HbA1c (Elevated Glucose/Diabetes)'),
                    'Hemoglobin': (12.0, None, 'Hemoglobin (Anemia/Low)'),
                    'Bicarbonate': (22.0, None, 'Bicarbonate (Metabolic Acidosis/Low)'),
                    'Albumin_Creatinine_Ratio': (None, 30.0, 'Albumin-Creatinine Ratio (High)'),
                    'Fasting_Glucose': (None, 100.0, 'Fasting Glucose (High)'),
                    'Cholesterol': (None, 200.0, 'Cholesterol (High)')
                }
                
                patient_factors = []
                for col, limits in clinical_ranges.items():
                    lower, upper, friendly = limits
                    val = patient_dict.get(col)
                    if val is not None and not pd.isna(val):
                        dev = 0.0
                        if upper is not None and val > upper:
                            dev = (val - upper) / (upper if upper != 0 else 1.0)
                        elif lower is not None and val < lower:
                            dev = (lower - val) / lower
                            
                        if dev > 0:
                            importance = base_importances.get(col, 0.05)
                            score = dev * importance
                            patient_factors.append({
                                'feature': col,
                                'name': friendly,
                                'contribution': float(score)
                            })
                
                patient_factors = sorted(patient_factors, key=lambda x: x['contribution'], reverse=True)
                risk_factors = patient_factors[:4]
        except Exception as e_rf:
            print(f"Error extracting risk factors: {e_rf}")

        return jsonify({
            'status': 'success',
            'prediction': int(pred),
            'label': ckd_stage,
            'probability': float(prob_ckd),
            'risk_level': risk_level,
            'gfr': float(gfr),
            'ckd_stage': ckd_stage,
            'risk_factors': risk_factors
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
