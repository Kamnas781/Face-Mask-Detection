import os
import time
import urllib.request
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "face mask detection data")
WITH_MASK_DIR = os.path.join(DATASET_DIR, "with_mask")
WITHOUT_MASK_DIR = os.path.join(DATASET_DIR, "without_mask")

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


face_cascade = cv2.CascadeClassifier(ensure_haar_cascade())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened. Check that your camera is connected.")

print("Auto capture mode")
print("Press 'w' to capture WITH mask photos automatically")
print("Press 'n' to capture WITHOUT mask photos automatically")
print("Press 'q' to quit")

count = {
    "with_mask": len([name for name in os.listdir(WITH_MASK_DIR) if name.lower().endswith((".png", ".jpg", ".jpeg"))]),
    "without_mask": len([name for name in os.listdir(WITHOUT_MASK_DIR) if name.lower().endswith((".png", ".jpg", ".jpeg"))]),
}
last_capture_time = 0
capture_interval = 0.5
capture_mode = None
capture_limit = {label: total + 30 for label, total in count.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if capture_mode is not None and time.time() - last_capture_time >= capture_interval:
        if len(faces) == 0:
            print("No face detected. Waiting...")
        else:
            (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
            face = frame[y:y + h, x:x + w]
            label_dir = WITH_MASK_DIR if capture_mode == "with_mask" else WITHOUT_MASK_DIR
            label_name = "with_mask" if capture_mode == "with_mask" else "without_mask"

            count[label_name] += 1
            filename = os.path.join(label_dir, f"{label_name}_{count[label_name]}.png")
            cv2.imwrite(filename, face)
            print(f"Saved {label_name} image #{count[label_name]}: {filename}")

            if count[label_name] >= capture_limit[label_name]:
                print(f"Captured 30 new images for {label_name}. Stopping capture.")
                capture_mode = None

        last_capture_time = time.time()

    cv2.putText(
        frame,
        "Press 'w' = auto capture WITH mask | 'n' = auto capture WITHOUT mask | 'q' = quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Auto Capture Face Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        capture_mode = "with_mask"
        count["with_mask"] = max(0, count["with_mask"])
        last_capture_time = 0
        print("Starting auto capture for WITH mask. Press 'q' to stop.")

    elif key == ord('n'):
        capture_mode = "without_mask"
        count["without_mask"] = max(0, count["without_mask"])
        last_capture_time = 0
        print("Starting auto capture for WITHOUT mask. Press 'q' to stop.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done. Images saved to:")
print(WITH_MASK_DIR)
print(WITHOUT_MASK_DIR)
