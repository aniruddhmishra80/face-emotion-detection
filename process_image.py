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

# Function to process uploaded image
def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (p, q, r, s) in faces:
        face = gray[q:q+s, p:p+r]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        cv2.rectangle(image, (p, q), (p+r, q+s), (255, 0, 0), 2)
        cv2.putText(image, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
    cv2.imshow("Uploaded Image Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# File dialog for image upload
Tk().withdraw()  # Hide the root window
image_path = filedialog.askopenfilename(title="Select an Image File")
if image_path:
    process_image(image_path)