import argparse
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Train the face mask classifier.")
parser.add_argument(
    "--dataset",
    default="dataset",
    help="Folder containing with_mask and without_mask subfolders (default: dataset).",
)
dataset_argument = parser.parse_args().dataset
DATASET_DIR = (
    dataset_argument
    if os.path.isabs(dataset_argument)
    else os.path.join(BASE_DIR, dataset_argument)
)
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"The dataset folder is missing at '{DATASET_DIR}'. Create it with subfolders like 'with_mask' and 'without_mask' before training."
    )

for class_name in ("with_mask", "without_mask"):
    class_dir = os.path.join(DATASET_DIR, class_name)
    image_count = sum(
        name.lower().endswith((".png", ".jpg", ".jpeg"))
        for name in os.listdir(class_dir)
    ) if os.path.isdir(class_dir) else 0
    if image_count < 5:
        raise ValueError(
            f"Dataset '{DATASET_DIR}' needs at least 5 labeled images in '{class_name}'. "
            "Check the labels and save more correctly labeled face images before training."
        )

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.2,
)
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    classes=["with_mask", "without_mask"],
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
)

validation_data = validation_datagen.flow_from_directory(
    DATASET_DIR,
    classes=["with_mask", "without_mask"],
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=10,
        restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    ),
]

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

final_val_acc = max(history.history["val_accuracy"]) if history.history.get("val_accuracy") else history.history["accuracy"][-1]
print(f"Best validation accuracy: {final_val_acc:.4f}")

if final_val_acc < 0.95:
    print("Validation accuracy is below 0.95. Unfreezing the base model for fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=30,
        callbacks=[
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=8,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
            ),
        ],
        verbose=1,
    )

    final_val_acc = max(fine_tune_history.history["val_accuracy"]) if fine_tune_history.history.get("val_accuracy") else fine_tune_history.history["accuracy"][-1]
    print(f"Fine-tuned validation accuracy: {final_val_acc:.4f}")

model.save(MODEL_PATH)
print(f"Model trained and saved successfully to: {MODEL_PATH}")