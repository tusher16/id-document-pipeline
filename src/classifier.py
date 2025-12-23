from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except Exception as e:
    tf = None
    layers = models = callbacks = None
    _TF_IMPORT_ERR = e


@dataclass
class ClassifierConfig:
    img_size: Tuple[int, int] = (512, 512)
    threshold: float = 0.5


class IDCardClassifier:
    """
    Binary classifier: ID-like vs not-ID.
    Train expects a folder structure like:
      Train/
        class0/
        class1/
      Test/
        class0/
        class1/
    (same as your course setup)
    """
    def __init__(self, cfg: ClassifierConfig = ClassifierConfig()):
        if tf is None:
            raise ImportError(f"TensorFlow not available: {_TF_IMPORT_ERR}")
        self.cfg = cfg
        self.model: Optional[tf.keras.Model] = None

    def build(self) -> "IDCardClassifier":
        model = models.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(*self.cfg.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model = model
        return self

    def train_from_basepath(self, base_path: str | Path, epochs: int = 10, batch_size: int = 32):
        base_path = Path(base_path)
        train_dir = base_path / "1_classification" / "Train"
        test_dir  = base_path / "1_classification" / "Test"

        datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_gen = datagen_train.flow_from_directory(
            str(train_dir), target_size=self.cfg.img_size, batch_size=batch_size, class_mode="binary"
        )
        test_gen = datagen_test.flow_from_directory(
            str(test_dir), target_size=self.cfg.img_size, batch_size=batch_size, class_mode="binary"
        )

        if self.model is None:
            self.build()

        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
        ]

        hist = self.model.fit(train_gen, validation_data=test_gen, epochs=epochs, callbacks=cb)
        return hist

    def save(self, path: str | Path):
        if self.model is None:
            raise ValueError("Model not built/loaded.")
        self.model.save(str(path))

    def load(self, path: str | Path) -> "IDCardClassifier":
        self.model = tf.keras.models.load_model(str(path))
        return self

    def predict_proba(self, bgr: np.ndarray) -> float:
        if self.model is None:
            raise ValueError("Model not built/loaded.")
        img = cv2.resize(bgr, self.cfg.img_size)
        img = img.astype(np.float32) / 255.0
        pred = self.model.predict(img[None, ...], verbose=0)[0][0]
        return float(pred)

    def predict_label(self, bgr: np.ndarray) -> int:
        return int(self.predict_proba(bgr) >= self.cfg.threshold)
