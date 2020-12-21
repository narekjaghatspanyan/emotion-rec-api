from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np

def detect_emotion(image):
        detection_model_path = 'haarcascade_frontalface_default.xml'
        emotion_model_path = 'emotion_detection_model.hdf5'
        EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        face_detection = cv2.CascadeClassifier(detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)
        face_image = cv2.imread(image)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
                faces = sorted(faces, reverse=True,
                               key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        # emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        # print(label)
        return str(label)