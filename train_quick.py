import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import cv2

TRAIN_DIR = 'images/images/train'
TEST_DIR = 'images/images/test'

def extract_features(images):
    features = []
    for image in images:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features / 255.0

print('Loading training data...')
train_image_paths = []
train_labels = []
for label in os.listdir(TRAIN_DIR):
    for imagename in os.listdir(os.path.join(TRAIN_DIR, label))[0:100]: # just use 100 images per class for speed
        train_image_paths.append(os.path.join(TRAIN_DIR, label, imagename))
        train_labels.append(label)

print('Loading testing data...')
test_image_paths = []
test_labels = []
for label in os.listdir(TEST_DIR):
    for imagename in os.listdir(os.path.join(TEST_DIR, label))[0:20]: # just use 20 images per class
        test_image_paths.append(os.path.join(TEST_DIR, label, imagename))
        test_labels.append(label)

print('Extracting features...')    
x_train = extract_features(train_image_paths)
x_test = extract_features(test_image_paths)

label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
y_train_int = [label_map[l] for l in train_labels]
y_test_int = [label_map[l] for l in test_labels]

y_train = to_categorical(y_train_int, num_classes=7)
y_test = to_categorical(y_test_int, num_classes=7)

print('Building model...')
model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Training for 1 epoch to generate the h5 file...')
model.fit(x=x_train, y=y_train, batch_size=128, epochs=1, validation_data=(x_test,y_test))

model_json = model.to_json()
with open('emotiondetector.json', 'w') as json_file:
    json_file.write(model_json)
model.save('emotiondetector.h5')
print('Saved emotiondetector.h5 and emotiondetector.json')
