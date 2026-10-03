import os
import base64
import binascii
import ipaddress
import re
import shutil
import threading
from datetime import datetime
from io import BytesIO
from urllib.parse import urlsplit

import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image, UnidentifiedImageError

from reinforcement_learning import FeedbackLearner, FeedbackTokenError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mask_detector.h5")
DATASET_DIR = os.path.join(BASE_DIR, "face mask detection data")
FEEDBACK_POLICY_PATH = os.path.join(DATASET_DIR, "mask_feedback_policy.json")
LATEST_IMAGE_PATH = os.path.join(DATASET_DIR, "latest_detection.jpg")
network_camera_lock = threading.Lock()
network_camera_capture = None
auto_capture_lock = threading.Lock()
auto_capture_dataset_name = "face mask detection data"
latest_image_lock = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: '{MODEL_PATH}'. Train the model first using train_model.py."
    )

model = tf.keras.models.load_model(MODEL_PATH)
feedback_learner = FeedbackLearner(FEEDBACK_POLICY_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def predict_mask_from_image(image):
    input_height, input_width = model.input_shape[1:3]
    model_image = image.resize((input_width or 224, input_height or 224))
    image_array = np.asarray(model_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    preview_buffer = BytesIO()
    image.save(preview_buffer, format="JPEG", quality=88)
    image_data_url = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.getvalue()).decode("ascii")

    prediction = model.predict(image_array, verbose=0)[0][0]
    if prediction >= 0.5:
        label = "Without Mask"
        confidence = float(prediction)
    else:
        label = "With Mask"
        confidence = float(1.0 - prediction)

    return label, confidence, image_data_url


def custom_dataset_dir(dataset_name):
    if not isinstance(dataset_name, str):
        raise ValueError("Enter a dataset folder name.")

    dataset_name = dataset_name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,39}", dataset_name):
        raise ValueError("Use 1-40 letters, numbers, spaces, hyphens, or underscores.")

    return os.path.join(DATASET_DIR, "user_datasets", dataset_name)


def saved_image_timestamp(filename, image_path):
    timestamp_match = re.search(r"_(\d{8}_\d{6}_\d{6})$", os.path.splitext(filename)[0])
    if timestamp_match:
        try:
            return datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S_%f")
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(image_path))


def validate_network_camera_url(stream_url):
    if not isinstance(stream_url, str) or not stream_url or len(stream_url) > 2048:
        raise ValueError("Enter a valid camera stream address.")
    if any(character.isspace() or ord(character) < 32 for character in stream_url):
        raise ValueError("The camera stream address cannot contain spaces or control characters.")

    try:
        parsed_url = urlsplit(stream_url)
        hostname = parsed_url.hostname
        port = parsed_url.port
    except ValueError as exc:
        raise ValueError("Enter a valid camera stream address.") from exc

    if (
        parsed_url.scheme.lower() not in {"rtsp", "rtsps", "http", "https"}
        or not hostname
        or parsed_url.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("Use an RTSP, RTSPS, HTTP, or HTTPS camera stream address.")

    hostname = hostname.rstrip(".").lower()
    if hostname.endswith(".local"):
        return stream_url

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as exc:
        raise ValueError(
            "For privacy, use the camera's private network IP address or a .local hostname."
        ) from exc
    if not address.is_private or address.is_multicast or address.is_unspecified:
        raise ValueError("For privacy, connect only to a camera on your private network.")
    return stream_url


def saved_dataset_roots():
    roots = {}
    custom_root = os.path.join(DATASET_DIR, "user_datasets")
    if os.path.isdir(custom_root):
        for name in sorted(os.listdir(custom_root), key=str.casefold):
            try:
                root = custom_dataset_dir(name)
            except ValueError:
                continue
            if os.path.isdir(root):
                roots[f"custom-{name}"] = (name, root)
    return roots


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/saved-data")
def saved_data():
    roots = saved_dataset_roots()
    selected_key = request.args.get("dataset")
    if selected_key not in roots:
        selected_key = next(iter(roots), None)

    dataset_dir = roots[selected_key][1] if selected_key is not None else None
    classes = []
    for class_name in ("with_mask", "without_mask"):
        class_dir = os.path.join(dataset_dir, class_name) if dataset_dir else None
        filenames = []
        if class_dir and os.path.isdir(class_dir):
            filenames = sorted(
                (
                    name for name in os.listdir(class_dir)
                    if name.lower().endswith((".png", ".jpg", ".jpeg"))
                    and os.path.isfile(os.path.join(class_dir, name))
                ),
                key=str.casefold,
                reverse=True,
            )
        classes.append({
            "name": class_name,
            "count": len(filenames),
            "images": [
                {
                    "name": filename,
                    "timestamp": saved_image_timestamp(
                        filename,
                        os.path.join(class_dir, filename),
                    ),
                    "url": url_for(
                        "saved_data_image",
                        dataset_key=selected_key,
                        class_name=class_name,
                        filename=filename,
                    ),
                }
                for filename in filenames
            ],
        })

    return render_template(
        "saved_data.html",
        datasets=[{"key": key, "name": value[0]} for key, value in roots.items()],
        selected_key=selected_key,
        selected_dataset_name=roots[selected_key][0] if selected_key is not None else None,
        classes=classes,
    )


@app.get("/saved-data/image/<dataset_key>/<class_name>/<filename>")
def saved_data_image(dataset_key, class_name, filename):
    if class_name not in {"with_mask", "without_mask"}:
        abort(404)

    root = saved_dataset_roots().get(dataset_key)
    if root is None:
        abort(404)

    class_dir = os.path.join(root[1], class_name)
    if not os.path.isfile(os.path.join(class_dir, filename)):
        abort(404)
    return send_from_directory(class_dir, filename)


@app.get("/saved-data/latest-image")
def saved_latest_image():
    with latest_image_lock:
        if not os.path.isfile(LATEST_IMAGE_PATH):
            abort(404)
    return send_from_directory(DATASET_DIR, os.path.basename(LATEST_IMAGE_PATH))


@app.post("/saved-data/latest-image/clear")
def clear_latest_image():
    with latest_image_lock:
        if not os.path.isfile(LATEST_IMAGE_PATH):
            return jsonify({"error": "There is no current detection image to clear."}), 404
        if os.path.islink(LATEST_IMAGE_PATH):
            return jsonify({"error": "The current detection image cannot be cleared because it is a symbolic link."}), 400
        os.unlink(LATEST_IMAGE_PATH)
    return jsonify({"cleared": True})


@app.post("/network-camera/connect")
def connect_network_camera():
    global network_camera_capture

    payload = request.get_json(silent=True) or {}
    try:
        stream_url = validate_network_camera_url(payload.get("stream_url"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    candidate = cv2.VideoCapture()
    try:
        opened = candidate.open(
            stream_url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ],
        )
        if not opened:
            candidate.release()
            return jsonify({"error": "Could not open the camera stream. Check its address, credentials, and network connection."}), 400

        success, frame = candidate.read()
        if not success or frame is None:
            candidate.release()
            return jsonify({"error": "The camera connected but did not provide a video frame."}), 400
    except cv2.error:
        candidate.release()
        return jsonify({"error": "Could not open this camera stream. Check the address and supported stream format."}), 400

    with network_camera_lock:
        previous_capture = network_camera_capture
        network_camera_capture = candidate
        if previous_capture is not None:
            previous_capture.release()

    return jsonify({"connected": True})


@app.get("/network-camera/frame")
def network_camera_frame():
    global network_camera_capture

    with network_camera_lock:
        if network_camera_capture is None:
            return jsonify({"error": "No network camera is connected."}), 409
        success, frame = network_camera_capture.read()
        if not success or frame is None:
            network_camera_capture.release()
            network_camera_capture = None
            return jsonify({"error": "The network camera stopped sending video. Reconnect and try again."}), 503

    success, encoded_frame = cv2.imencode(".jpg", frame)
    if not success:
        return jsonify({"error": "Could not encode a frame from the network camera."}), 500
    return Response(encoded_frame.tobytes(), mimetype="image/jpeg")


@app.post("/network-camera/stop")
def stop_network_camera():
    global network_camera_capture

    with network_camera_lock:
        if network_camera_capture is not None:
            network_camera_capture.release()
            network_camera_capture = None
    return jsonify({"connected": False})


@app.post("/create-dataset")
def create_dataset():
    payload = request.get_json(silent=True) or {}
    try:
        dataset_dir = custom_dataset_dir(payload.get("dataset_name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    os.makedirs(os.path.join(dataset_dir, "with_mask"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "without_mask"), exist_ok=True)
    return jsonify({
        "dataset_name": os.path.basename(dataset_dir),
        "dataset_path": os.path.relpath(dataset_dir, BASE_DIR),
    })


@app.post("/delete-dataset")
def delete_dataset():
    payload = request.get_json(silent=True) or {}
    try:
        dataset_dir = custom_dataset_dir(payload.get("dataset_name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if os.path.islink(dataset_dir):
        return jsonify({"error": "Dataset folders that are symbolic links cannot be deleted here."}), 400
    if not os.path.isdir(dataset_dir):
        return jsonify({"error": "That dataset folder does not exist."}), 404

    dataset_root = os.path.realpath(os.path.join(DATASET_DIR, "user_datasets"))
    resolved_dataset_dir = os.path.realpath(dataset_dir)
    if os.path.commonpath((dataset_root, resolved_dataset_dir)) != dataset_root:
        return jsonify({"error": "The requested folder is outside the dataset directory."}), 400

    shutil.rmtree(dataset_dir)
    return jsonify({"dataset_name": os.path.basename(dataset_dir)})


@app.post("/clear-dataset")
def clear_dataset():
    payload = request.get_json(silent=True) or {}
    try:
        dataset_dir = custom_dataset_dir(payload.get("dataset_name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    dataset_root = os.path.realpath(os.path.join(DATASET_DIR, "user_datasets"))
    if os.path.islink(dataset_dir) or not os.path.isdir(dataset_dir):
        return jsonify({"error": "That dataset folder does not exist or cannot be cleared."}), 404

    resolved_dataset_dir = os.path.realpath(dataset_dir)
    if os.path.commonpath((dataset_root, resolved_dataset_dir)) != dataset_root:
        return jsonify({"error": "The requested folder is outside the dataset directory."}), 400

    deleted_count = 0
    image_extensions = (".png", ".jpg", ".jpeg")
    for class_name in ("with_mask", "without_mask"):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.islink(class_dir) or not os.path.isdir(class_dir):
            continue
        for entry in os.scandir(class_dir):
            if (
                entry.name.lower().endswith(image_extensions)
                and entry.is_file(follow_symlinks=False)
            ):
                os.unlink(entry.path)
                deleted_count += 1

    return jsonify({
        "dataset_name": os.path.basename(dataset_dir),
        "deleted_count": deleted_count,
    })


@app.post("/delete-selected-data")
def delete_selected_data():
    payload = request.get_json(silent=True) or {}
    try:
        dataset_dir = custom_dataset_dir(payload.get("dataset_name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    selected_images = payload.get("images")
    if (
        not isinstance(selected_images, list)
        or not selected_images
        or len(selected_images) > 500
    ):
        return jsonify({"error": "Select between 1 and 500 images to delete."}), 400

    dataset_root = os.path.realpath(os.path.join(DATASET_DIR, "user_datasets"))
    if os.path.islink(dataset_dir) or not os.path.isdir(dataset_dir):
        return jsonify({"error": "That dataset folder does not exist or cannot be changed."}), 404

    resolved_dataset_dir = os.path.realpath(dataset_dir)
    if os.path.commonpath((dataset_root, resolved_dataset_dir)) != dataset_root:
        return jsonify({"error": "The requested folder is outside the dataset directory."}), 400

    image_extensions = (".png", ".jpg", ".jpeg")
    paths_to_delete = []
    seen_paths = set()
    for image in selected_images:
        if not isinstance(image, dict):
            return jsonify({"error": "Each selected image must include its class and filename."}), 400

        class_name = image.get("class_name")
        filename = image.get("filename")
        if (
            class_name not in {"with_mask", "without_mask"}
            or not isinstance(filename, str)
            or os.path.basename(filename) != filename
            or not filename.lower().endswith(image_extensions)
        ):
            return jsonify({"error": "A selected image has an invalid class or filename."}), 400

        class_dir = os.path.join(dataset_dir, class_name)
        image_path = os.path.join(class_dir, filename)
        if os.path.islink(class_dir) or not os.path.isdir(class_dir):
            return jsonify({"error": "A selected image folder does not exist or cannot be changed."}), 404
        if os.path.islink(image_path) or not os.path.isfile(image_path):
            return jsonify({"error": f"Selected image '{filename}' was not found."}), 404

        resolved_class_dir = os.path.realpath(class_dir)
        if os.path.commonpath((resolved_dataset_dir, resolved_class_dir)) != resolved_dataset_dir:
            return jsonify({"error": "A selected image is outside the dataset folder."}), 400
        if image_path not in seen_paths:
            paths_to_delete.append(image_path)
            seen_paths.add(image_path)

    for image_path in paths_to_delete:
        os.unlink(image_path)

    return jsonify({
        "dataset_name": os.path.basename(dataset_dir),
        "deleted_count": len(paths_to_delete),
    })


@app.post("/capture")
def capture():
    payload = request.get_json(silent=True) or {}
    save_to_dataset = payload.get("save_to_dataset") is True
    save_detection = payload.get("save_detection") is True
    dataset_dir = None
    dataset_label = payload.get("dataset_label")
    if save_to_dataset:
        try:
            dataset_dir = custom_dataset_dir(payload.get("dataset_name"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if dataset_label not in {"with_mask", "without_mask"}:
            return jsonify({"error": "Choose the correct mask label before saving training samples."}), 400
        if not all(
            os.path.isdir(os.path.join(dataset_dir, class_name))
            for class_name in ("with_mask", "without_mask")
        ):
            return jsonify({"error": "Create the dataset folder before saving samples."}), 400

    image_data = payload.get("image_data", "")
    prefix = "data:image/jpeg;base64,"
    if not isinstance(image_data, str) or not image_data.startswith(prefix):
        return jsonify({"error": "Capture a webcam image before analyzing."}), 400

    try:
        image_bytes = base64.b64decode(image_data[len(prefix):], validate=True)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "Could not read the captured image. Please try again."}), 400

    frame = np.asarray(image)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )
    if len(faces) == 0:
        return jsonify({"face_detected": False, "message": "No face detected. Face the camera and try again."})

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    face_image = image.crop((x, y, x + width, y + height))
    os.makedirs(DATASET_DIR, exist_ok=True)
    with latest_image_lock:
        face_image.save(LATEST_IMAGE_PATH, format="JPEG", quality=90)
    model_label, confidence, preview = predict_mask_from_image(face_image)
    model_label_key = "with_mask" if model_label == "With Mask" else "without_mask"
    feedback = feedback_learner.register_prediction(model_label_key, confidence)
    label = "With Mask" if feedback["label"] == "with_mask" else "Without Mask"
    dataset_saved_path = None
    dataset_saved_label = None
    if dataset_dir is not None:
        class_dir = os.path.join(dataset_dir, dataset_label)
        filename = f"{dataset_label}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        dataset_path = os.path.join(class_dir, filename)
        face_image.save(dataset_path, format="JPEG", quality=90)
        dataset_saved_path = os.path.relpath(dataset_path, BASE_DIR)
        dataset_saved_label = dataset_label
    elif save_detection:
        with auto_capture_lock:
            dataset_dir = custom_dataset_dir(auto_capture_dataset_name)
            if os.path.islink(dataset_dir) or not os.path.isdir(dataset_dir):
                return jsonify({
                    "error": (
                        f"Create the '{auto_capture_dataset_name}' dataset folder "
                        "before enabling automatic saving."
                    )
                }), 409
            dataset_saved_label = "with_mask" if label == "With Mask" else "without_mask"
            class_dir = os.path.join(dataset_dir, dataset_saved_label)
            if os.path.islink(class_dir) or not os.path.isdir(class_dir):
                return jsonify({
                    "error": (
                        f"The '{dataset_saved_label}' folder is missing from "
                        f"'{auto_capture_dataset_name}'. Recreate the dataset folder."
                    )
                }), 409
            filename = f"{dataset_saved_label}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            dataset_path = os.path.join(class_dir, filename)
            face_image.save(dataset_path, format="JPEG", quality=90)
            dataset_saved_path = os.path.relpath(dataset_path, BASE_DIR)

    return jsonify({
        "face_detected": True,
        "label": label,
        "model_label": model_label,
        "learner_adjusted": feedback["adjusted"],
        "feedback_id": feedback["feedback_id"],
        "feedback_state_count": feedback["state_feedback_count"],
        "confidence": confidence,
        "image_data_url": preview,
        "saved_path": os.path.relpath(LATEST_IMAGE_PATH, BASE_DIR),
        "dataset_saved_path": dataset_saved_path,
        "dataset_saved_label": dataset_saved_label,
    })


@app.post("/feedback")
def submit_feedback():
    payload = request.get_json(silent=True) or {}
    actual_label = payload.get("actual_label")
    feedback_id = payload.get("feedback_id")
    try:
        result = feedback_learner.record_feedback(feedback_id, actual_label)
    except FeedbackTokenError as exc:
        return jsonify({"error": str(exc)}), 410
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError:
        app.logger.exception("Could not save reinforcement-learning feedback.")
        return jsonify({"error": "Feedback was not recorded because the learner could not save its policy."}), 500

    return jsonify(result)


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return render_template("index.html", result={"label": "No image uploaded", "confidence": 0.0})

    image_file = request.files["image"]
    if image_file.filename == "":
        return render_template("index.html", result={"label": "No image selected", "confidence": 0.0})

    try:
        image = Image.open(image_file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return render_template(
            "index.html",
            result={"label": "Please upload a valid image", "confidence": 0.0},
        )

    label, confidence, image_data_url = predict_mask_from_image(image)
    return render_template(
        "index.html",
        result={
            "label": label,
            "confidence": confidence,
            "image_data_url": image_data_url,
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
