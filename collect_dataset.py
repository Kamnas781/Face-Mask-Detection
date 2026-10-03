import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "face mask detection data")
WITH_MASK_DIR = os.path.join(DATASET_DIR, "with_mask")
WITHOUT_MASK_DIR = os.path.join(DATASET_DIR, "without_mask")

os.makedirs(WITH_MASK_DIR, exist_ok=True)
os.makedirs(WITHOUT_MASK_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found. Please connect your camera and try again.")

print("Instructions:")
print("- Press 'w' to capture a WITH MASK face")
print("- Press 'n' to capture a WITHOUT MASK face")
print("- Press 'q' to quit")

counts = {
    "with_mask": len([name for name in os.listdir(WITH_MASK_DIR) if name.lower().endswith((".png", ".jpg", ".jpeg"))]),
    "without_mask": len([name for name in os.listdir(WITHOUT_MASK_DIR) if name.lower().endswith((".png", ".jpg", ".jpeg"))]),
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read webcam frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        "Press 'w' = with mask | 'n' = without mask | 'q' = quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Collect Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        if len(faces) == 0:
            print("No face detected. Move closer to the camera and try again.")
            continue

        (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
        face = frame[y:y+h, x:x+w]
        counts["with_mask"] += 1
        filename = os.path.join(WITH_MASK_DIR, f"with_mask_{counts['with_mask']}.png")
        cv2.imwrite(filename, face)
        print(f"Saved with mask image: {filename}")

    elif key == ord('n'):
        if len(faces) == 0:
            print("No face detected. Move closer to the camera and try again.")
            continue

        (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
        face = frame[y:y+h, x:x+w]
        counts["without_mask"] += 1
        filename = os.path.join(WITHOUT_MASK_DIR, f"without_mask_{counts['without_mask']}.png")
        cv2.imwrite(filename, face)
        print(f"Saved without mask image: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset collection complete.")
print("Folders:")
print(WITH_MASK_DIR)
print(WITHOUT_MASK_DIR)
