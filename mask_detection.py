import os
import subprocess
import sys
import urllib.request
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

if not os.path.exists(MODEL_PATH):
    TRAIN_SCRIPT = os.path.join(BASE_DIR, "train.py")
    if not os.path.isfile(TRAIN_SCRIPT):
        raise FileNotFoundError(f"Training script was not found: '{TRAIN_SCRIPT}'")

    print("Trained model not found. Training a model now; this may take a few minutes...")
    subprocess.run([sys.executable, TRAIN_SCRIPT], cwd=BASE_DIR, check=True)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Training finished without creating the expected model file: '{MODEL_PATH}'"
        )


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


# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load face detector
face_cascade = cv2.CascadeClassifier(ensure_haar_cascade())

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check that your camera is connected and available.")

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Process every detected face
    for (x, y, w, h) in faces:

        # Extract face
        face = frame[y:y+h, x:x+w]

        # Match the image dimensions used when the model was trained.
        input_height, input_width = model.input_shape[1:3]
        face = cv2.resize(face, (input_width or 224, input_height or 224))

        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize
        face = face / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Predict
        prediction = model.predict(face, verbose=0)[0][0]

        # Keras class order is with_mask=0, without_mask=1.
        if prediction >= 0.5:
            label = "No Mask"
            confidence = prediction
        else:
            label = "Mask"
            confidence = 1 - prediction

        # Display text
        text = f"{label}: {confidence * 100:.2f}%"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    # Display webcam
    cv2.imshow("Face Mask Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()