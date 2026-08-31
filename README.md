# Smart Home Access Control

Streamlit deployment package for the Smart Home Access Control academic prototype.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
Set the main file to `app.py` and use Python 3.12.

## Model
- HOG feature extraction
- RBF SVM classifier with probability estimates
- Authorization threshold: 0.65
- Model labels: `0 = Unknown_Visitors`, `1 = Authorized_Residents`
- Face detection: OpenCV Haar Cascade

The trained model file is `smart_home_access_control.joblib`.

> Demo/academic prototype only. The supplied dataset is very small and this system should not be used as the sole mechanism for real-world physical access decisions.
