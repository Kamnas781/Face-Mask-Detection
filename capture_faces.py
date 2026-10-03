import os
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

print("Instructions:")
print("1. Press 'w' to save a face WITH mask")
print("2. Press 'n' to save a face WITHOUT mask")
print("3. Press 'q' to quit")

count = {
    "with_mask": len([name for name in os.listdir(WITH_MASK_DIR) if name.lower().endswith((".png", ".jpg", ".jpeg"))]),
    "without_mask": len([name for name in os.listdir(WITHOUT_MASK_DIR) if name.lower().endswith((".png", ".jpg", ".jpeg"))]),
}

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

    cv2.putText(
        frame,
        "Press 'w' = with mask | 'n' = without mask | 'q' = quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Capture Face Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        if len(faces) == 0:
            print("No face detected. Please move closer to the camera and try again.")
            continue
        (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
        face = frame[y:y + h, x:x + w]
        count["with_mask"] += 1
        filename = os.path.join(WITH_MASK_DIR, f"with_mask_{count['with_mask']}.png")
        cv2.imwrite(filename, face)
        print(f"Saved face with mask: {filename}")

    elif key == ord('n'):
        if len(faces) == 0:
            print("No face detected. Please move closer to the camera and try again.")
            continue
        (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
        face = frame[y:y + h, x:x + w]
        count["without_mask"] += 1
        filename = os.path.join(WITHOUT_MASK_DIR, f"without_mask_{count['without_mask']}.png")
        cv2.imwrite(filename, face)
        print(f"Saved face without mask: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done. Images saved to:")
print(WITH_MASK_DIR)
print(WITHOUT_MASK_DIR)
