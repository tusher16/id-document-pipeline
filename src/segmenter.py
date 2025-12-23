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
class SegmenterConfig:
    img_size: Tuple[int, int] = (512, 512)
    threshold: float = 0.5


class IDCardSegmenter:
    """
    U-Net segmentation.
    Expects dataset layout from your course:
      2_segmentation/Train/Ids
      2_segmentation/Train/GroundTruth
      2_segmentation/Test/Ids
      2_segmentation/Test/GroundTruth
    """
    def __init__(self, cfg: SegmenterConfig = SegmenterConfig()):
        if tf is None:
            raise ImportError(f"TensorFlow not available: {_TF_IMPORT_ERR}")
        self.cfg = cfg
        self.model: Optional[tf.keras.Model] = None

    def _conv_block(self, x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return x

    def build(self) -> "IDCardSegmenter":
        inputs = layers.Input(shape=(*self.cfg.img_size, 3))

        c1 = self._conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
        c2 = self._conv_block(p1, 64);     p2 = layers.MaxPooling2D()(c2)
        c3 = self._conv_block(p2, 128);    p3 = layers.MaxPooling2D()(c3)
        c4 = self._conv_block(p3, 256);    p4 = layers.MaxPooling2D()(c4)

        bn = self._conv_block(p4, 512)

        u1 = layers.UpSampling2D()(bn); u1 = layers.Concatenate()([u1, c4]); c5 = self._conv_block(u1, 256)
        u2 = layers.UpSampling2D()(c5); u2 = layers.Concatenate()([u2, c3]); c6 = self._conv_block(u2, 128)
        u3 = layers.UpSampling2D()(c6); u3 = layers.Concatenate()([u3, c2]); c7 = self._conv_block(u3, 64)
        u4 = layers.UpSampling2D()(c7); u4 = layers.Concatenate()([u4, c1]); c8 = self._conv_block(u4, 32)

        outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model = model
        return self

    @staticmethod
    def _load_pair(img_path, mask_path, img_size):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, img_size)
        mask = tf.cast(mask, tf.float32) / 255.0
        return img, mask

    def train_from_basepath(self, base_path: str | Path, epochs: int = 10, batch_size: int = 4):
        base_path = Path(base_path)
        tr_img = base_path / "2_segmentation" / "Train" / "Ids"
        tr_msk = base_path / "2_segmentation" / "Train" / "GroundTruth"
        va_img = base_path / "2_segmentation" / "Test" / "Ids"
        va_msk = base_path / "2_segmentation" / "Test" / "GroundTruth"

        train_imgs = sorted([str(p) for p in tr_img.glob("*.png")])
        train_msks = sorted([str(p) for p in tr_msk.glob("*.png")])
        val_imgs   = sorted([str(p) for p in va_img.glob("*.png")])
        val_msks   = sorted([str(p) for p in va_msk.glob("*.png")])

        train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_msks))
        train_ds = train_ds.map(lambda a,b: self._load_pair(a,b,self.cfg.img_size), num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_msks))
        val_ds = val_ds.map(lambda a,b: self._load_pair(a,b,self.cfg.img_size), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        if self.model is None:
            self.build()

        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
        ]
        hist = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
        return hist

    def save(self, path: str | Path):
        if self.model is None:
            raise ValueError("Model not built/loaded.")
        self.model.save(str(path))

    def load(self, path: str | Path) -> "IDCardSegmenter":
        self.model = tf.keras.models.load_model(str(path))
        return self

    def predict_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Returns float mask [0..1] in model resolution."""
        if self.model is None:
            raise ValueError("Model not built/loaded.")
        img = cv2.resize(bgr, self.cfg.img_size)
        img = img.astype(np.float32) / 255.0
        pred = self.model.predict(img[None, ...], verbose=0)[0, ..., 0]
        return pred

    def mask_to_bbox(self, mask01: np.ndarray) -> tuple[int,int,int,int] | None:
        """Threshold + find largest contour bbox in mask coords."""
        m = (mask01 >= self.cfg.threshold).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return x,y,w,h

    def crop_original_with_mask_bbox(self, bgr: np.ndarray, bbox_in_mask_coords: tuple[int,int,int,int]) -> np.ndarray:
        """Map bbox from mask-size to original image and crop."""
        x,y,w,h = bbox_in_mask_coords
        H, W = bgr.shape[:2]
        mh, mw = self.cfg.img_size
        sx, sy = W / mw, H / mh
        X = int(x * sx); Y = int(y * sy)
        WW = int(w * sx); HH = int(h * sy)
        X = max(0, X); Y = max(0, Y)
        return bgr[Y:Y+HH, X:X+WW]
