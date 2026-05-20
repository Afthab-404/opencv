import face_recognition
import cv2
import numpy as np

# Open webcam
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load image
user_image = face_recognition.load_image_file("user/passportsize.jpg")
user_face_encoding = face_recognition.face_encodings(user_image)[0]

# Known faces
known_face_encodings = [
    user_face_encoding
]

known_face_names = [
    "user"
]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    # Read frame
    ret, frame = video_capture.read()

    if not ret:
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

    # Process every alternate frame
    if process_this_frame:

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

    process_this_frame = not process_this_frame

    # Draw boxes
    for (top, right, bottom, left), name in zip(
        face_locations,
        face_names
    ):

        # Scale back
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Draw label
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            (0, 255, 0),
            cv2.FILLED
        )

        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            1
        )

    # Show output
    cv2.imshow("Face Recognition", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()