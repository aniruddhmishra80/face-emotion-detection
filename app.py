import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
import os
import tempfile

# Configure the Streamlit page
st.set_page_config(page_title="Real-Time Face Emotion Detection", page_icon="🎭", layout="centered")

# Load model cache to prevent reloading on every interaction
@st.cache_resource
def load_model():
    try:
        json_file = open("emotiondetector.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("emotiondetector.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load Haar Cascade
@st.cache_resource
def load_cascade():
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(haar_file)

model = load_model()
face_cascade = load_cascade()

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def process_frame(frame):
    if model is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(frame, (p, q), (p+r, q+s), (0, 255, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img, verbose=0)
        prediction_label = labels[pred.argmax()]
        confidence = pred.max() * 100

        text = f"{prediction_label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

st.title("🎭 Real-Time Face Emotion Detection")
st.markdown("A pretrained CNN-based model to detect 7 human emotions (angry, disgust, fear, happy, neutral, sad, surprise).")

if model is None:
    st.error("Model could not be loaded. Ensure 'emotiondetector.json' and 'emotiondetector.h5' exist.")
    st.stop()

option = st.sidebar.selectbox("Select Input Source", ["Webcam", "Image Upload", "Video Upload"])

if option == "Webcam":
    st.header("Webcam Feed")
    st.markdown("Check the box below to start the webcam. Uncheck to stop.")
    run = st.checkbox("Start Webcam")
    frame_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to access webcam.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Streamlit expects RGB
                processed_frame = process_frame(frame)
                frame_placeholder.image(processed_frame, channels="RGB")
                
            cap.release()

elif option == "Image Upload":
    st.header("Image Processing")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        processed_image = process_frame(image)
        st.image(processed_image, channels="RGB", use_container_width=True)

elif option == "Video Upload":
    st.header("Video Processing")
    uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        
        stop_button = st.button("Stop Video")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_frame(frame)
            frame_placeholder.image(processed_frame, channels="RGB")
            
        cap.release()
        os.unlink(tfile.name)
