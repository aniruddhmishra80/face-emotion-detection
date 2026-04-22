import cv2
from keras.models import model_from_json
import numpy as np
from tkinter import filedialog
from tkinter import Tk

# Load the model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (p, q, r, s) in faces:
            face = gray[q:q+s, p:p+r]
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow("Uploaded Video Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

# File dialog for video upload
Tk().withdraw()  # Hide the root window
video_path = filedialog.askopenfilename(title="Select a Video File")
if video_path:
    process_video(video_path)