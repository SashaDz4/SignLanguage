import cv2
from model import LetterDetector, Letter
from utils import preparation_data, put_ukranian_labels
import time


video_path = ""
cap = cv2.VideoCapture(video_path)

model = LetterDetector((63, 1))
model.load('models/model.h5')
letter = Letter("", 0.0)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        count += 1
        if count % 10 == 0:
            start = time.time()
            letter, img = model.predict_classes(frame)
            end = time.time()
            print(f"Detected letter: {letter}, time: {end - start}")
        frame = put_ukranian_labels(frame, letter.name, letter.confidence)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Prediction", frame)
    if cv2.waitKey(32) & 0xFF == ord('q'):
        break

