import os
import urllib.request
import numpy as np
import cv2
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "face mask detection data")
WITH_MASK_DIR = os.path.join(DATASET_DIR, "with_mask")
WITHOUT_MASK_DIR = os.path.join(DATASET_DIR, "without_mask")
LATEST_IMAGE_PATH = os.path.join(DATASET_DIR, "latest_detection.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "mask_detector.h5")

os.makedirs(WITH_MASK_DIR, exist_ok=True)
os.makedirs(WITHOUT_MASK_DIR, exist_ok=True)


def ensure_haar_cascade():
    cascade_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
    cascade_name = "haarcascade_frontalface_default.xml"
    cascade_path = os.path.join(cascade_dir, cascade_name)

    if os.path.exists(cascade_path):
        return cascade_path

    os.makedirs(cascade_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        with urllib.request.urlopen(url, timeout=20) as response, open(cascade_path, "wb") as file:
            file.write(response.read())
        return cascade_path
    except Exception as exc:
        raise FileNotFoundError(
            "OpenCV face cascade file is missing and could not be downloaded. Please reinstall OpenCV or check your internet connection."
        ) from exc


def open_webcam():
    backends = (
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
    )
    attempts = []

    for camera_index in (0, 1):
        for backend_name, backend in backends:
            camera = cv2.VideoCapture(camera_index, backend)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, frame = camera.read()
                if ret and frame is not None:
                    print(f"Webcam opened using {backend_name} (device {camera_index}).")
                    return camera, frame

            camera.release()
            attempts.append(f"device {camera_index} ({backend_name})")

    tried = ", ".join(attempts)
    raise RuntimeError(
        f"Could not read a frame from the webcam. Tried {tried}. "
        "Close other apps using the camera, check Windows camera permissions, "
        "and reconnect the webcam."
    )


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' was not found. Run train_model.py first to train and save the model."
    )

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(ensure_haar_cascade())

cap, first_frame = open_webcam()

print("Automatic mask detection dataset collection started.")
print("The model will decide if the face is WITH MASK or WITHOUT MASK.")
print("Press 'q' to stop early.")

latest_label = "Waiting for face"
latest_confidence = 0.0

while True:
    if first_frame is not None:
        frame = first_frame
        first_frame = None
    else:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
        face = frame[y:y + h, x:x + w]
        input_height, input_width = model.input_shape[1:3]
        face = cv2.resize(face, (input_width or 224, input_height or 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)[0][0]
        is_mask = prediction < 0.5
        label = "with_mask" if is_mask else "without_mask"
        latest_label = "WITH MASK" if is_mask else "WITHOUT MASK"
        latest_confidence = prediction if is_mask else 1.0 - prediction
        cv2.imwrite(LATEST_IMAGE_PATH, frame[y:y + h, x:x + w])

        color = (0, 255, 0) if is_mask else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, latest_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    status = f"Latest: {latest_label} | confidence: {latest_confidence:.1%}"
    cv2.putText(
        frame,
        status,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Press 'q' to stop",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imshow("Auto Mask Dataset Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Latest detection image saved to:")
print(LATEST_IMAGE_PATH)
