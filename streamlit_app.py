"""
Handwritten Alphabet Recognition - Streamlit App (Live Webcam Version)
Innomatics Research Labs - Image Classification Capstone Project

This version supports TWO input modes:
    1. Upload an image (your original behaviour)
    2. Live webcam air-writing, running directly in the browser using
       streamlit-webrtc, so it also works on the deployed Streamlit Cloud
       website (not just locally).

Why streamlit-webrtc?
    cv2.VideoCapture(0) (used in mediapipe_app.py) only works locally,
    because it accesses the webcam device on the machine actually running
    the Python process. On Streamlit Cloud, that machine is a remote
    server with no camera. streamlit-webrtc instead streams the VISITOR'S
    browser webcam frame-by-frame into your Python code, which is what
    lets this work on the deployed website.

Run this app using:
    streamlit run streamlit_app.py

Required packages (add to requirements.txt):
    streamlit
    streamlit-webrtc
    av
    opencv-python-headless
    mediapipe
    joblib
    numpy
    scikit-image
"""

import os
import time
import threading
import urllib.request

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from skimage.feature import hog
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# ---------------------------------------------------------
# Settings (same as mediapipe_app.py)
# ---------------------------------------------------------
CANVAS_SIZE = 400
TRAINING_IMAGE_SIZE = 34
LINE_THICKNESS = 8
PINCH_THRESHOLD = 0.06

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

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


def get_letter_bounding_box(canvas, padding=20):
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min = max(0, xs.min() - padding)
    x_max = min(canvas.shape[1], xs.max() + padding)
    y_min = max(0, ys.min() - padding)
    y_max = min(canvas.shape[0], ys.max() + padding)
    return x_min, x_max, y_min, y_max


def predict_from_canvas(canvas):
    """Used by the live webcam tab -- canvas is the black/white drawing, not a photo."""
    bbox = get_letter_bounding_box(canvas)
    if bbox is None:
        return None, 0.0

    x_min, x_max, y_min, y_max = bbox
    cropped = canvas[y_min:y_max, x_min:x_max]

    h, w = cropped.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    resized = cv2.resize(square, (TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    features = extract_features(resized).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_letter = label_encoder.inverse_transform([prediction])[0]
    confidence = round(max(probabilities) * 100, 2)
    return predicted_letter, confidence


# ---------------------------------------------------------
# Live webcam video processor -- this runs once per frame from the
# visitor's browser webcam, streamed in through streamlit-webrtc.
# ---------------------------------------------------------
class HandWritingProcessor(VideoProcessorBase):
    def __init__(self):
        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.prev_point = None
        self.start_time = time.time()

        # Flags set from the main Streamlit thread by button clicks
        self.clear_requested = False
        self.predict_requested = False

        # Results read back by the main Streamlit thread
        self.last_letter = ""
        self.last_confidence = 0.0

        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        cam_h, cam_w = img.shape[:2]

        with self.lock:
            if self.clear_requested:
                self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
                self.prev_point = None
                self.last_letter = ""
                self.last_confidence = 0.0
                self.clear_requested = False

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        fingertip_point = None
        is_pinching = False

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            x_px = int(index_tip.x * cam_w)
            y_px = int(index_tip.y * cam_h)

            canvas_x = int(index_tip.x * CANVAS_SIZE)
            canvas_y = int(index_tip.y * CANVAS_SIZE)
            fingertip_point = (canvas_x, canvas_y)

            pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            is_pinching = pinch_distance < PINCH_THRESHOLD

            dot_color = (0, 255, 0) if is_pinching else (0, 0, 255)
            cv2.circle(img, (x_px, y_px), 8, dot_color, -1)

        with self.lock:
            if fingertip_point is not None and is_pinching:
                if self.prev_point is not None:
                    cv2.line(self.canvas, self.prev_point, fingertip_point, 255, LINE_THICKNESS)
                self.prev_point = fingertip_point
            else:
                self.prev_point = None

            if self.predict_requested:
                letter, confidence = predict_from_canvas(self.canvas)
                if letter is not None:
                    self.last_letter, self.last_confidence = letter, confidence
                self.predict_requested = False

            canvas_copy = self.canvas.copy()
            last_letter = self.last_letter
            last_confidence = self.last_confidence

        # Overlay a small canvas preview + prediction text onto the video frame
        preview_size = 150
        preview = cv2.resize(canvas_copy, (preview_size, preview_size))
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        # Always show a cursor marking exactly where the fingertip currently
        # maps to on the canvas -- even while the pen is "up" (not pinching).
        # Without this, it's hard to line up separate strokes (like the two
        # legs of an "A" or the three bars of an "E") with what's already
        # drawn, because you can't see where your finger is relative to the
        # canvas until you start drawing again.
        if fingertip_point is not None:
            preview_x = int(np.clip(fingertip_point[0] * preview_size / CANVAS_SIZE, 0, preview_size - 1))
            preview_y = int(np.clip(fingertip_point[1] * preview_size / CANVAS_SIZE, 0, preview_size - 1))
            cursor_color = (0, 255, 0) if is_pinching else (0, 0, 255)
            cv2.drawMarker(preview_bgr, (preview_x, preview_y), cursor_color,
                            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        img[10:10 + preview_size, cam_w - (preview_size + 10):cam_w - 10] = preview_bgr
        cv2.rectangle(img, (cam_w - (preview_size + 10), 10), (cam_w - 10, 10 + preview_size), (255, 255, 255), 2)

        cv2.putText(img, "pinch thumb+index to draw", (10, cam_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if last_letter:
            cv2.putText(img, f"Prediction: {last_letter} ({last_confidence}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("Handwritten Alphabet Recognition")

tab_upload, tab_live = st.tabs(["Upload Image", "Live Webcam (Air-Writing)"])

with tab_upload:
    st.write(
        "Upload an image of a single handwritten letter (a white stroke on a "
        "black background works best, matching the training data style)."
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

with tab_live:
    st.write(
        "Allow camera access below, then **pinch your thumb and index finger "
        "together** to draw a letter in the air. Release the pinch to "
        "reposition without drawing."
    )

    ctx = webrtc_streamer(
        key="air-writing",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_processor_factory=HandWritingProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Canvas", key="clear_live"):
            if ctx.video_processor:
                ctx.video_processor.clear_requested = True
    with col2:
        if st.button("Predict Letter", key="predict_live"):
            if ctx.video_processor:
                ctx.video_processor.predict_requested = True
    with col3:
        if st.button("Refresh Result", key="refresh_live"):
            st.rerun()

    st.caption(
        "The prediction also appears directly on the video feed. If you don't "
        "see it update below right after clicking 'Predict Letter', click "
        "'Refresh Result'."
    )

    if ctx.video_processor and ctx.video_processor.last_letter:
        st.success(
            f"Predicted Letter: **{ctx.video_processor.last_letter}** "
            f"({ctx.video_processor.last_confidence}% confidence)"
        )

st.markdown("---")
st.caption("Image Classification Capstone Project - Innomatics Research Labs")
