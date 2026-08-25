# Chronic Kidney Disease Diagnosis using Machine Learning (NephroAI)

This repository contains the source code and documentation for the **NephroAI** project—a machine learning-based clinical decision support system for predicting Chronic Kidney Disease (CKD) and classifying its severity stages (Stages 1–5).

Created by: **Donipudi Asish Kumar**

## Repository Structure & Documentation

* **[`src/`](file:///c:/Users/donip/Desktop/CKD%20diagnosing%20using%20ML/src/)**: Model training, cleaning, and predictive pipelines.
* **[`backend/`](file:///c:/Users/donip/Desktop/CKD%20diagnosing%20using%20ML/backend/)**: Flask REST API serving the predictions, explainable risk factors, and eGFR calculations.
* **[`frontend/`](file:///c:/Users/donip/Desktop/CKD%20diagnosing%20using%20ML/frontend/)**: Vanilla HTML5/CSS3/JavaScript clinical dashboard.
* **[`data/`](file:///c:/Users/donip/Desktop/CKD%20diagnosing%20using%20ML/data/)**: Datasets (cleaned and raw).
* **[`models/`](file:///c:/Users/donip/Desktop/CKD%20diagnosing%20using%20ML/models/)**: Serialized models.

### Key Project Documentation

* **[Methodology (`methodology.txt`)](file:///c:/Users/donip/Desktop/CKD%20diagnosing%20using%20ML/methodology.txt)**: End-to-end overview of research objectives, preprocessing, candidate models, performance, and validation testing.
* **[Feature Extraction and Engineering (`feature_extraction.txt`)](file:///c:/Users/donip/Desktop/CKD%20diagnosing%20using%20ML/feature_extraction.txt)**: Comprehensive documentation detailing the 34 input features, preprocessing steps, leakage mitigation (eGFR removal), clinical eGFR calculation formula, and Explainable AI (XAI) feature contribution extraction.
* **[Tools and Libraries (`tools_and_libraries.txt`)](file:///c:/Users/donip/Desktop/CKD%20diagnosing%20using%20ML/tools_and_libraries.txt)**: Full details on all programming languages, standard library modules, and external data science/ML packages used in this project.
