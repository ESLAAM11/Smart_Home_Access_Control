import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog

MODEL_PATH = "smart_home_access_control.joblib"


def build_face_detector():
    """Create the OpenCV Haar detector when available.
    If OpenCV is present but the cascade API/data is unavailable in the
    deployment environment, return None and let the app classify the full image.
    """
    try:
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if cascade_cls is None or haar_dir is None:
            return None
        cascade_path = haar_dir + "haarcascade_frontalface_alt2.xml"
        detector = cascade_cls(cascade_path)
        if detector.empty():
            return None
        return detector
    except Exception:
        return None


FACE = build_face_detector()

st.set_page_config(
    page_title="Smart Home Access Control",
    page_icon="🏠",
    layout="centered",
)
st.title("🏠 Smart Home Access Control")
st.caption("AI / Machine Learning prototype — HOG + RBF SVM")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def extract_hog(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features.reshape(1, -1)


bundle = load_model()
clf = bundle["model"]
default_threshold = float(bundle.get("threshold", 0.65))
threshold = st.sidebar.slider(
    "Authorization threshold",
    0.50,
    0.95,
    default_threshold,
    0.01,
)

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    annotated = img.copy()
    results = []

    faces = []
    if FACE is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE.detectMultiScale(gray, 1.05, 3, minSize=(25, 25))

    if len(faces) == 0:
        # Compatibility fallback: classify the full image when no face is found
        # or when the cloud environment lacks the OpenCV Haar cascade API.
        feat = extract_hog(img)
        p = float(clf.predict_proba(feat)[0][1])
        decision = "Authorized_Residents" if p >= threshold else "Unknown_Visitors"
        results.append(("full image", decision, p))
    else:
        for i, (x, y, w, h) in enumerate(faces, 1):
            pad = int(0.35 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            crop = img[y1:y2, x1:x2]

            feat = extract_hog(crop)
            p = float(clf.predict_proba(feat)[0][1])
            decision = "Authorized_Residents" if p >= threshold else "Unknown_Visitors"
            results.append((f"face {i}", decision, p))

            box_color = (0, 255, 0) if decision.startswith("Authorized") else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                annotated,
                decision,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Access-control analysis")
    for region, decision, p in results:
        if decision == "Authorized_Residents":
            st.success(f"🔓 ACCESS GRANTED — {region} — probability: {p:.3f}")
        else:
            st.error(f"🔒 ACCESS DENIED — {region} — probability: {p:.3f}")

st.markdown("---")
st.warning(
    "Prototype only: the supplied dataset is very small. "
    "Do not use this as the sole mechanism for real-world physical access decisions."
)
