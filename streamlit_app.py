"""
Handwritten Alphabet Recognition - Streamlit App (Upload-Only Deployed Version)
Innomatics Research Labs - Image Classification Capstone Project

This is the DEPLOYED version -- upload an image only. No live webcam.

Why no live webcam here?
    The live air-writing feature (see mediapipe_app.py) needs mediapipe +
    OpenCV + streamlit-webrtc to stream a visitor's browser camera into the
    app and run real-time hand tracking. On Streamlit Community Cloud this
    repeatedly hit two separate platform limitations:
      1. mediapipe/OpenCV need system-level graphics libraries (libGL etc.)
         that are awkward to keep working reliably on Cloud's Linux image.
      2. Cloud's outbound network restricts the UDP traffic that WebRTC
         needs to establish a live video connection, so even with the
         libraries fixed, the connection itself was unreliable.
    Rather than fight those platform limits indefinitely, this deployed
    version keeps only the Upload Image tab, which needs none of that --
    no cv2, no mediapipe, no streamlit-webrtc. For the full live
    air-writing experience, run mediapipe_app.py locally (see README).

Run this app using:
    streamlit run streamlit_app.py

Required packages (see requirements.txt):
    streamlit
    joblib
    numpy
    scikit-image
    scikit-learn
    pillow
"""

import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog

# ---------------------------------------------------------
# Settings (same as mediapipe_app.py)
# ---------------------------------------------------------
TRAINING_IMAGE_SIZE = 34

# ---------------------------------------------------------
# Cached resources -- loaded once per server process, not per user session
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    model = joblib.load("trained_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder


model, label_encoder = load_ml_model()


def extract_features(image_34x34):
    img_array = np.array(image_34x34) / 255.0
    features = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def predict_from_pil_image(image):
    """Used by the upload-a-file tab."""
    image = image.convert("L").resize((TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE))
    features = extract_features(image).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_letter = label_encoder.inverse_transform([prediction])[0]
    confidence = round(max(probabilities) * 100, 2)

    prob_dict = {
        label_encoder.classes_[i]: round(probabilities[i] * 100, 2)
        for i in range(len(label_encoder.classes_))
    }
    return predicted_letter, confidence, prob_dict


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("Handwritten Alphabet Recognition")

st.write(
    "Upload an image of a single handwritten letter (a white stroke on a "
    "black background works best, matching the training data style)."
)
st.caption(
    "Tip: for a live demo of the air-writing feature (drawing letters in "
    "the air with your fingertip), see the project README -- it runs "
    "locally via mediapipe_app.py."
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if st.button("Predict", key="predict_upload"):
        letter, confidence, prob_dict = predict_from_pil_image(image)

        st.subheader("Prediction Result")
        st.success(f"Predicted Letter: **{letter}**")
        st.write(f"**Confidence Score:** {confidence}%")

        st.subheader("Top 5 Predictions")
        top_5 = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5])
        st.bar_chart(top_5)

st.markdown("---")
st.caption("Image Classification Capstone Project - Innomatics Research Labs")
