import cv2
from model import LetterDetector
from utils import preparation_data, put_ukranian_labels
import time

# path_to_image = "/home/oleksandr/SignLanguage/data/dataset/validation/0/2.jpg"
path_to_image = "/home/oleksandr/Downloads/test_img_E.jpg"

model = LetterDetector((63, 1))
model.load("models/model_my.h5")

input_IMG = cv2.imread(path_to_image)

# Print the Prediction
letter = model.predict_classes(input_IMG)
print("Prediction : ", letter)

input_IMG = cv2.resize(input_IMG, (640, 480))
input_IMG = put_ukranian_labels(input_IMG, letter.name, letter.confidence)
cv2.imshow("Prediction", input_IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()

video_path = "/home/oleksandr/Downloads/Telegram Desktop/IMG_0367.MOV"
cap = cv2.VideoCapture(video_path)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        count += 1
        if count % 10 == 0:
            start = time.time()
            letter = model.predict_classes(frame)
            end = time.time()
            print(f"Detected letter: {letter}, time: {end - start}")
        frame = put_ukranian_labels(frame, letter.name, letter.confidence)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Prediction", frame)
    if cv2.waitKey(32) & 0xFF == ord('q'):
        break

