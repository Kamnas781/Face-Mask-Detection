# Face Mask Detection

This project includes a TensorFlow/Keras face-mask classifier, a webcam
dataset collector, and a Flask/webcam application for running predictions.
The browser app supports local and authorized private-network cameras,
timestamped face samples, dataset management, and optional user feedback that
adapts future predictions.

## Features

- Live **With Mask / Without Mask** detection from a local webcam.
- Optional RTSP/RTSPS or HTTP/HTTPS IP-camera input on a private network.
- Automatic saving of one image per continuous face detection. Another image
  is saved after the face has been absent for three consecutive checks and is
  detected again.
- Saved images include date and time in the filename and are sorted into
  `with_mask` and `without_mask` under
  `face mask detection data/user_datasets/face mask detection data/`.
- The **View saved data** page browses samples, shows timestamps, and lets you
  delete selected images or clear all images in the selected dataset.
- User feedback can confirm or correct a prediction. Feedback is kept locally
  and may adjust later predictions after enough examples are collected.

## Dataset

Training images are organized by their true label:

```text
dataset/
  with_mask/
  without_mask/
```

Put labeled face images in the corresponding folders. The training script
requires at least five images in each class; use a larger, varied, and
reasonably balanced dataset for more reliable results.

The webcam collector `collect_dataset.py` saves labeled images into these
folders. The web app can also save images into named folders under
`face mask detection data/user_datasets/`.

## Install dependencies

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you do not use the included virtual environment, replace that interpreter
path with your Python executable.

The application expects the trained model file `mask_detector.h5` in the
project folder. If it is missing, train the model using the instructions
below.

## Train the model

Train using `dataset/`:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

To train using a named web-app dataset instead, pass its folder path. That
folder must contain `with_mask` and `without_mask` subfolders:

```powershell
.\.venv\Scripts\python.exe train_model.py --dataset "face mask detection data\user_datasets\My Dataset"
```

Training uses TensorFlow/Keras with a MobileNetV2 image-classification model
and saves the result as `mask_detector.h5` in the project folder. Restart the
Flask app after training so it loads the updated model.

## Run detection

Start the web app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000` in a browser. Select **Camera On** and allow
camera access when prompted. Use **Close Camera** to stop detection.

For the desktop webcam detector instead:

```powershell
.\.venv\Scripts\python.exe mask_detection.py
```

The app and webcam detector use the saved model's input dimensions when
preparing face images for predictions.

## Connect a Wi-Fi/IP camera

On the camera page, enter a camera stream address such as
`rtsp://192.168.1.50:554/stream` or
`http://192.168.1.50:8080/video`, then select **Connect network camera** and
confirm that you have permission to view that camera. The app supports RTSP,
RTSPS, and HTTP/HTTPS streams on a private network; use a private IP address
or a `.local` hostname. The app cannot trigger the browser's camera permission
prompt for an IP camera stream, so it asks for confirmation before connecting.
Any credentials embedded in the stream address are sent to this app for the
connection and are not saved. Use **Close Camera** to disconnect.

These automatically saved labels are predictions and may be wrong. Review
samples in **View saved data** before using them as training labels. Create
named dataset folders from the camera page; clear or delete a named folder
there, or use **Clear saved images** and **Delete selected** from Saved Data.

## Reinforcement-learning feedback

After a live face detection, use **Prediction is correct**, **With Mask**, or
**Without Mask** to give the learner feedback. This is a small contextual
reinforcement-learning policy layered on top of the classifier. The model's
label and confidence range form the state; the displayed label is the action;
confirming it gives a `+1` reward, while selecting the other label gives `-1`.
The policy waits for five reviewed examples in the same state before it can
adjust future displayed predictions. This feedback does **not** update or
retrain the TensorFlow model. Its compact policy is saved locally in
`face mask detection data/mask_feedback_policy.json`; the feedback endpoint
does not store additional face images.

## Troubleshooting

- **Camera permission:** allow camera access in the browser and open the app
  using `localhost` or a secure connection.
- **Model missing:** run the training command above to create
  `mask_detector.h5`.
- **Images not appearing:** verify that the dataset folder and its
  `with_mask`/`without_mask` class folders exist, then refresh **View saved
  data**.
- **Feedback does not change a prediction yet:** the learner needs at least
  five reviewed examples for the same predicted label and confidence range.
