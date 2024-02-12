from flask import Flask, render_template, Response
import cv2
import time

from model import LetterDetector, Letter
from utils import put_ukranian_labels

app = Flask(__name__)

video_path = ""
cap = cv2.VideoCapture(0)
model = LetterDetector((63, 1))
model.load('models/model.h5')


def generate_frames():
    count = 0
    letter = Letter("", 0.0)
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            count += 1
            if count % 1 == 0:
                letter, frame = model.predict_classes(frame)

                print(f"Detected letter: {letter}")
            frame = put_ukranian_labels(frame, letter.name, letter.confidence, treshold=0.7)
            # print fps
            cv2.putText(frame, f"FPS: {1 / (time.time() - start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            start = time.time()
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
