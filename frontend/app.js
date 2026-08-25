// Dynamic base URL supporting both local file system and hosted server routes
const API_BASE_URL = window.location.origin.startsWith('file') ? 'http://127.0.0.1:5000' : window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
    console.log('NephroAI Diagnostics Frontend initialized.');

    // DOM Elements
    const serverStatusDot = document.getElementById('server-status-dot');
    const serverStatusText = document.getElementById('server-status-text');
    const diagnosticForm = document.getElementById('diagnostic-form');
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsLoader = document.getElementById('results-loader');
    const resultsPredictionView = document.getElementById('results-prediction-view');
    const resultsPercentageValue = document.getElementById('results-percentage-value');
    const resultsGaugeFill = document.getElementById('results-gauge-fill');
    const resultsRiskBadge = document.getElementById('results-risk-badge');
    const resultsStatementText = document.getElementById('results-statement-text');
    const resultsFactorsContainer = document.getElementById('results-factors-container');
    const resultsFactorsSection = document.getElementById('results-factors-section');
    const btnReset = document.getElementById('btn-reset');

    // Presets
    const presetHealthyBtn = document.getElementById('preset-healthy-btn');
    const presetModerateBtn = document.getElementById('preset-moderate-btn');
    const presetSevereBtn = document.getElementById('preset-severe-btn');

    // Tab Navigation
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-content');

    // Form Input Fields Mapping (exactly matches backend/clean dataset casing)
    const inputs = {
        Age: document.getElementById('input-age'),
        Gender: document.getElementById('select-gender'),
        BMI: document.getElementById('input-bmi'),
        Systolic_BP: document.getElementById('input-systolic-bp'),
        Diastolic_BP: document.getElementById('input-diastolic-bp'),
        Heart_Rate: document.getElementById('input-heart-rate'),
        Urine_Specific_Gravity: document.getElementById('input-urine-specific-gravity'),
        Urine_Albumin: document.getElementById('input-urine-albumin'),
        Urine_Protein: document.getElementById('input-urine-protein'),
        Albumin_Creatinine_Ratio: document.getElementById('input-acr'),
        Serum_Creatinine: document.getElementById('input-sc'),
        Blood_Urea_Nitrogen: document.getElementById('input-bu'),
        Sodium: document.getElementById('input-sodium'),
        Potassium: document.getElementById('input-potassium'),
        Calcium: document.getElementById('input-calcium'),
        Phosphorus: document.getElementById('input-phosphorus'),
        Chloride: document.getElementById('input-chloride'),
        Bicarbonate: document.getElementById('input-bicarbonate'),
        Hemoglobin: document.getElementById('input-hemo'),
        RBC_Count: document.getElementById('input-rbcc'),
        WBC_Count: document.getElementById('input-wbcc'),
        Platelet_Count: document.getElementById('input-platelets'),
        Packed_Cell_Volume: document.getElementById('input-pcv'),
        Blood_Glucose_Random: document.getElementById('input-bgr'),
        Fasting_Glucose: document.getElementById('input-fasting-glucose'),
        HbA1c: document.getElementById('input-hba1c'),
        Cholesterol: document.getElementById('input-cholesterol'),
        Triglycerides: document.getElementById('input-triglycerides'),
        Serum_Albumin: document.getElementById('input-serum-albumin'),
        Total_Protein: document.getElementById('input-total-protein'),
        Diabetes: document.getElementById('select-dm'),
        Hypertension: document.getElementById('select-htn'),
        Smoking_Status: document.getElementById('select-smoking'),
        Family_History_Kidney: document.getElementById('select-family-history')
    };

    // Preset Patient Data Constants matching Cleaned Dataset columns
    const PRESETS = {
        healthy: {
            Age: 35, Gender: '0', BMI: 22, Systolic_BP: 110, Diastolic_BP: 70, Heart_Rate: 72,
            Urine_Specific_Gravity: 1.025, Urine_Albumin: 10, Urine_Protein: 0, Albumin_Creatinine_Ratio: 15,
            Serum_Creatinine: 0.7, Blood_Urea_Nitrogen: 12, Sodium: 140, Potassium: 4.0, Calcium: 9.5,
            Phosphorus: 3.5, Chloride: 101, Bicarbonate: 25, Hemoglobin: 14.5, RBC_Count: 4.9,
            WBC_Count: 6200, Platelet_Count: 260000, Packed_Cell_Volume: 43, Blood_Glucose_Random: 95,
            Fasting_Glucose: 85, HbA1c: 5.2, Cholesterol: 180, Triglycerides: 130, Serum_Albumin: 4.0,
            Total_Protein: 7.0, Diabetes: 'No', Hypertension: 'No', Smoking_Status: 'No', Family_History_Kidney: 'No'
        },
        moderate: {
            Age: 55, Gender: '1', BMI: 28, Systolic_BP: 138, Diastolic_BP: 85, Heart_Rate: 80,
            Urine_Specific_Gravity: 1.015, Urine_Albumin: 85, Urine_Protein: 30, Albumin_Creatinine_Ratio: 120,
            Serum_Creatinine: 1.8, Blood_Urea_Nitrogen: 35, Sodium: 137, Potassium: 4.8, Calcium: 8.8,
            Phosphorus: 4.2, Chloride: 104, Bicarbonate: 21, Hemoglobin: 11.2, RBC_Count: 4.1,
            WBC_Count: 7800, Platelet_Count: 210000, Packed_Cell_Volume: 33, Blood_Glucose_Random: 145,
            Fasting_Glucose: 112, HbA1c: 6.2, Cholesterol: 210, Triglycerides: 185, Serum_Albumin: 3.2,
            Total_Protein: 6.2, Diabetes: 'No', Hypertension: 'Yes', Smoking_Status: 'No', Family_History_Kidney: 'Yes'
        },
        severe: {
            Age: 65, Gender: '0', BMI: 30, Systolic_BP: 155, Diastolic_BP: 98, Heart_Rate: 88,
            Urine_Specific_Gravity: 1.008, Urine_Albumin: 320, Urine_Protein: 180, Albumin_Creatinine_Ratio: 480,
            Serum_Creatinine: 4.5, Blood_Urea_Nitrogen: 75, Sodium: 135, Potassium: 5.8, Calcium: 8.1,
            Phosphorus: 5.5, Chloride: 106, Bicarbonate: 16, Hemoglobin: 8.2, RBC_Count: 3.5,
            WBC_Count: 10200, Platelet_Count: 180000, Packed_Cell_Volume: 25, Blood_Glucose_Random: 195,
            Fasting_Glucose: 126, HbA1c: 7.8, Cholesterol: 245, Triglycerides: 250, Serum_Albumin: 2.2,
            Total_Protein: 5.6, Diabetes: 'Yes', Hypertension: 'Yes', Smoking_Status: 'Yes', Family_History_Kidney: 'Yes'
        }
    };

    // Check backend server connection status
    async function checkServerStatus() {
        try {
            console.log('Checking backend server connection...');
            const response = await fetch(`${API_BASE_URL}/health`);
            const data = await response.json();
            if (response.ok && data.status === 'healthy') {
                serverStatusDot.classList.add('online');
                serverStatusText.textContent = 'Server Connected';
                console.log('Backend server is ONLINE.');
                return true;
            }
        } catch (error) {
            console.warn('Backend server status check failed:', error);
        }
        serverStatusDot.classList.remove('online');
        serverStatusText.textContent = 'Server Offline';
        return false;
    }

    // Initial status check
    checkServerStatus();
    // Keep checking status every 10 seconds
    setInterval(checkServerStatus, 10000);

    // Tab navigation handler
    tabButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Deactivate all buttons & panels
            tabButtons.forEach(btn => {
                btn.classList.remove('active');
                btn.setAttribute('aria-selected', 'false');
            });
            tabPanels.forEach(panel => panel.classList.remove('active'));
            
            // Activate target button & panel
            button.classList.add('active');
            button.setAttribute('aria-selected', 'true');
            const targetPanelId = button.getAttribute('aria-controls');
            document.getElementById(targetPanelId).classList.add('active');
        });
    });

    // Load Preset Data
    function loadPreset(presetKey) {
        console.log(`Loading preset scenario: ${presetKey}`);
        const data = PRESETS[presetKey];
        if (!data) return;

        // Fill each input
        Object.keys(inputs).forEach(key => {
            if (inputs[key]) {
                inputs[key].value = data[key] !== undefined ? data[key] : '';
            }
        });

        // Provide visual feedback (fade in/out effect)
        const activePanel = document.querySelector('.tab-content.active');
        if (activePanel) {
            activePanel.style.opacity = '0.3';
            setTimeout(() => {
                activePanel.style.opacity = '1';
                activePanel.style.transition = 'opacity 0.3s ease';
            }, 150);
        }

        // Automatically trigger prediction run
        runDiagnosis();
    }

    presetHealthyBtn.addEventListener('click', () => loadPreset('healthy'));
    presetModerateBtn.addEventListener('click', () => loadPreset('moderate'));
    presetSevereBtn.addEventListener('click', () => loadPreset('severe'));

    // Reset Form Fields
    btnReset.addEventListener('click', () => {
        console.log('Resetting clinical input form.');
        diagnosticForm.reset();
        
        // Hide prediction results, show placeholder
        resultsPredictionView.classList.remove('active');
        resultsPlaceholder.style.display = 'flex';
    });

    // Collect all form input values into a JSON object
    function getFormValues() {
        const values = {};
        Object.keys(inputs).forEach(key => {
            const field = inputs[key];
            if (field) {
                const rawVal = field.value;
                if (rawVal === '' || rawVal === undefined) {
                    values[key] = '?';
                } else {
                    values[key] = rawVal;
                }
            }
        });
        return values;
    }

    // Run Diagnostic Prediction
    async function runDiagnosis() {
        console.log('Starting diagnostic prediction logic...');
        
        // Show loader, hide prediction and placeholder
        resultsPlaceholder.style.display = 'none';
        resultsPredictionView.classList.remove('active');
        resultsLoader.style.display = 'block';

        const payload = getFormValues();
        console.log('Collected patient payload:', payload);

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.message || 'Server returned an error');
            }

            const result = await response.json();
            console.log('Received API response:', result);
            
            if (result.status === 'success') {
                displayResults(result);
            } else {
                throw new Error(result.message || 'Unknown error occurred');
            }

        } catch (error) {
            console.error('Diagnosis prediction failed:', error);
            alert(`Prediction Error: ${error.message}\nMake sure the Flask backend server is running.`);
            resultsPlaceholder.style.display = 'flex';
            resultsPredictionView.classList.remove('active');
        } finally {
            resultsLoader.style.display = 'none';
        }
    }

    // Display results and animate gauge & factors
    function displayResults(data) {
        const probability = data.probability; // Float between 0 and 1
        const percentage = Math.round(probability * 100);
        
        // Show Prediction view
        resultsPredictionView.classList.add('active');
        resultsPlaceholder.style.display = 'none';

        // Animate Circular Gauge
        const circumference = 502;
        const offset = circumference - (probability * circumference);
        resultsGaugeFill.style.strokeDashoffset = offset;
        
        // Classify styling class based on staging prediction
        let color = 'var(--accent-green)';
        let riskClass = 'low';
        
        if (data.prediction === 0) {
            color = 'var(--accent-green)';
            riskClass = 'low';
        } else if (data.prediction === 1 || data.prediction === 2) {
            color = 'var(--accent-orange)';
            riskClass = 'moderate';
        } else {
            color = 'var(--accent-red)';
            riskClass = 'high';
        }

        resultsGaugeFill.style.stroke = color;
        resultsPercentageValue.textContent = `${percentage}%`;
        resultsPercentageValue.style.color = color;

        // Update Badge text
        resultsRiskBadge.className = `result-badge ${riskClass}`;
        resultsRiskBadge.textContent = data.label; // Stage label (e.g. "Moderate CKD (Stage 3)")

        // Update Statement based on GFR and prediction
        const gfrVal = data.gfr !== null ? `${Math.round(data.gfr)} mL/min/1.73m²` : 'Unavailable';
        
        if (data.prediction > 0) {
            resultsStatementText.innerHTML = `Our clinical model detects indicators of <strong>${data.label}</strong>. The patient's estimated GFR is <strong>${gfrVal}</strong> (Calculated via 2021 CKD-EPI) with an overall model risk score of <strong>${percentage}%</strong>. Consultation with a nephrologist and confirmatory renal function panels are recommended.`;
        } else {
            resultsStatementText.innerHTML = `No indicators of Chronic Kidney Disease were detected. The patient's estimated GFR is <strong>${gfrVal}</strong> with a minimal model risk probability of <strong>${percentage}%</strong>.`;
        }

        // Render Risk Factors
        resultsFactorsContainer.innerHTML = '';
        
        if (data.risk_factors && data.risk_factors.length > 0) {
            resultsFactorsSection.style.display = 'block';
            const maxContribution = Math.max(...data.risk_factors.map(f => f.contribution));

            data.risk_factors.forEach(factor => {
                const barWidth = maxContribution > 0 ? (factor.contribution / maxContribution) * 100 : 0;
                
                const factorEl = document.createElement('div');
                factorEl.className = 'factor-item';
                factorEl.innerHTML = `
                    <div class="factor-header">
                        <span class="factor-name">${factor.name}</span>
                        <span class="factor-score">+${(factor.contribution * 100).toFixed(1)}%</span>
                    </div>
                    <div class="factor-bar-bg">
                        <div class="factor-bar-fill" style="width: 0%; background: ${color}"></div>
                    </div>
                `;
                resultsFactorsContainer.appendChild(factorEl);

                // Animate bar loading
                setTimeout(() => {
                    const fillBar = factorEl.querySelector('.factor-bar-fill');
                    if (fillBar) {
                        fillBar.style.width = `${barWidth}%`;
                    }
                }, 100);
            });
        } else {
            resultsFactorsSection.style.display = 'none';
        }
    }

    // Intercept submit button click directly for maximum reliability
    const btnSubmit = document.getElementById('btn-submit');
    if (btnSubmit) {
        btnSubmit.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Submit button clicked directly.');
            runDiagnosis();
        });
    }

    // Intercept form submit
    diagnosticForm.addEventListener('submit', (e) => {
        e.preventDefault();
        console.log('Form submit intercepted.');
        runDiagnosis();
    });
});
