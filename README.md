# PatchSight-ImmuneMap-LifeSpan-PIL-Framework
PatchSight–ImmuneMap–LifeSpan is a unified AI framework for breast cancer analysis integrating histopathology classification, immune cell detection, and survival prediction. It combines optimized InceptionResNetV2, Faster R-CNN, and survival models to deliver accurate, interpretable, end-to-end clinical decision support.
## Key Features
- Unified three-phase AI pipeline for breast cancer analysis. 
- Patch-based histopathology classification using optimized InceptionResNetV2.  
- Robust benign vs malignant detection across multi-magnification images.
- Automated tumor-infiltrating lymphocyte (TIL) detection using Faster R-CNN.
- Immunoscore generation from IHC images (CD3/CD8)
## Applications
- Automated breast cancer diagnosis from histopathology images.
- Computer-aided pathology to support pathologists and reduce manual workload.
- Quantification of tumor-infiltrating lymphocytes from IHC slides
- Immunoscore-based assessment of tumor immune microenvironment
- Personalized breast cancer prognosis and survival prediction.
## Prerequisites 
- **Python 3.7 or higher**
- **Required libraries**: numpy pandas scikit-learn matplotlib scipy fuzzy / skfuzzy tensorflow / keras joblib, matplotlib, seaborn, optuna,shap, captum
- **Deep Learning Framework**: PyTorch 2.0+ (or TensorFlow equivalent if adapted)
- **Hardware Requirements:** GPU-enabled system (e.g., NVIDIA RTX 3090/4090 or equivalent). Minimum 16 GB RAM (recommended: 32 GB+)
## Evaluation Metrics
- **Classification Metrics**: 
  Accuracy (%) – Measures overall correctness of recurrence type prediction.
  Precision (%) – Indicates reliability of positive predictions (reduces false positives).
  Recall (%) – Measures the model’s ability to detect true recurrence cases (reduces false negatives).
  F1-Score (%) – Harmonic mean of precision and recall for balanced evaluation.
  AUC-ROC – Evaluates class separability across recurrence risk categories.
- **Survival Analysis Metrics**: 
  C-Index (Concordance Index) – Quantifies how well predicted survival times align with actual outcomes.
  Kaplan–Meier (KM) Curves – Visualize survival probability over time for risk groups.
  Log-Rank Test (p-value) – Tests statistical significance between high- and low-risk survival groups.
  Brier Score – Measures calibration accuracy of predicted survival probabilities.Time-Dependent AUC (10-year AUC).
## How to Run
1. Clone the Repository
git clone https://github.com/<your-username>/PatchSight-ImmuneMap-LifeSpan-PIL-Framework.git  
cd PatchSight-ImmuneMap-LifeSpan-PIL-Framework

2. Set Up a Python Environment
Create and activate a new environment (recommended):
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install Dependencies
Install all required libraries:
pip install -r requirements.txt

4. Download Datasets
Download and organize datasets used in the study:


5. Preprocess Data
Run preprocessing to clean, normalize, and prepare datasets:
python scripts/preprocess_data.py
This step handles:
Missing value imputation
Normalization (Z-score scaling)
Feature encoding
Autoencoder-based dimensionality reduction

6. Train the Model
Train the unified multi-modal Transformer:
python train_model.py
This script:
Loads preprocessed data
Trains autoencoders and Transformer encoder
Optimizes classification, survival, reconstruction, and contrastive losses
Saves the best model under /models/checkpoints/
Training parameters can be adjusted in config.yaml (batch size, epochs, learning rate, etc.).

7. Evaluate the Model
After training, evaluate performance on the test sets:
python evaluate_model.py
Outputs include:
Accuracy, Precision, Recall, F1-score
C-index, Kaplan–Meier curves, Log-Rank test
SHAP explainability plots

8. Visualize and Interpret Results
Generate interpretability and visualization plots:
python visualize_results.py
This script creates:
SHAP-based feature importance charts
Kaplan–Meier survival plots
Training vs. validation accuracy/loss graphs

9. Predict New Patient Data
To predict recurrence risk and survival probability for a new patient:
python predict_patient.py --input patient_data.csv
Output includes predicted recurrence category, log-risk score, and survival probability.

10. Optional: Hyperparameter Optimization
To optimize model parameters:
python tune_hyperparameters.py
Uses Optuna for cross-validation tuning across multiple datasets.


  
