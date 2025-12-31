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
