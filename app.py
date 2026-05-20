from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

# Open webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load image
user_image = face_recognition.load_image_file(
    "user/passportsize.jpg"
)

user_face_encoding = face_recognition.face_encodings(
    user_image
)[0]

# Known encodings
known_face_encodings = [
    user_face_encoding
]

known_face_names = [
    "user"
]

def gen_frames():

    while True:

        success, frame = camera.read()

        if not success:
            continue

        # Resize frame
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            face_locations
        )

        face_names = []

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding
            )

            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Draw face boxes
        for (top, right, bottom, left), name in zip(
            face_locations,
            face_names
        ):

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Convert frame to bytes
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(debug=True)