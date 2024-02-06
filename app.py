from flask import Flask, render_template, Response
import cv2

from model import LetterDetector, Letter
from utils import put_ukranian_labels

app = Flask(__name__)

video_path = "/home/oleksandr/Downloads/Telegram Desktop/IMG_0367.MOV"
cap = cv2.VideoCapture(video_path)
model = LetterDetector((63, 1))
model.load('models/model1.h5')


def generate_frames():
    count = 0
    letter = Letter("", 0.8)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            count += 1
            if count % 10 == 0:
                letter = model.predict_classes(frame)
                print(f"Detected letter: {letter}")
            frame = put_ukranian_labels(frame, letter.name, letter.confidence)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
