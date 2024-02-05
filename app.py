from flask import Flask, render_template, Response
import cv2

from model import LetterDetector

app = Flask(__name__)

cap = cv2.VideoCapture(0)
model = LetterDetector((63, 1))
model.load('models/model1.h5')


def generate_frames():
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            count += 1
            if count % 10 == 0:
                prediction = model.predict_classes(frame)
                cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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