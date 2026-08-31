import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog

MODEL_PATH = 'smart_home_access_control.joblib'
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

st.set_page_config(page_title='Smart Home Access Control', page_icon='🏠', layout='centered')
st.title('🏠 Smart Home Access Control')
st.caption('AI / Machine Learning prototype — HOG + RBF SVM')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()
clf = bundle['model']
default_threshold = float(bundle.get('threshold', 0.65))
threshold = st.sidebar.slider('Authorization threshold', 0.50, 0.95, default_threshold, 0.01)

uploaded = st.file_uploader('Upload a photo', type=['jpg', 'jpeg', 'png'])

if uploaded:
    pil = Image.open(uploaded).convert('RGB')
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE.detectMultiScale(gray, 1.05, 3, minSize=(25, 25))

    annotated = img.copy()
    results = []
    if len(faces) == 0:
        # Keep compatibility with the supplied prototype: classify the full image.
        crop = img
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.resize(gray_crop, (128, 128), interpolation=cv2.INTER_AREA)
        feat = hog(gray_crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)
        p = float(clf.predict_proba(feat)[0][1])
        decision = 'Authorized_Residents' if p >= threshold else 'Unknown_Visitors'
        results.append(('full image', decision, p))
    else:
        for i, (x, y, w, h) in enumerate(faces, 1):
            pad = int(0.35 * max(w, h))
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
            crop = img[y1:y2, x1:x2]
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
            feat = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)
            p = float(clf.predict_proba(feat)[0][1])
            decision = 'Authorized_Residents' if p >= threshold else 'Unknown_Visitors'
            results.append((f'face {i}', decision, p))
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0) if decision.startswith('Authorized') else (0, 0, 255), 2)
            cv2.putText(annotated, decision, (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption='Access-control analysis')
    for region, decision, p in results:
        if decision == 'Authorized_Residents':
            st.success(f'🔓 ACCESS GRANTED — {region} — probability: {p:.3f}')
        else:
            st.error(f'🔒 ACCESS DENIED — {region} — probability: {p:.3f}')

st.markdown('---')
st.warning('Prototype only: the supplied dataset is very small. Do not use this as the sole mechanism for real-world physical access decisions.')
