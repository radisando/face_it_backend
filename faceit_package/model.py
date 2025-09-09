# faceit_package/model.py
from dotenv import load_dotenv
load_dotenv(override=True)

import os, io, json, typing
from typing import Optional, Tuple, Union, Dict, List
import numpy as np
from PIL import Image, ImageOps

# Prefer Keras 3; fall back to tf.keras
try:
    import keras  # Keras 3
    from keras.saving import register_keras_serializable
    from keras.applications.resnet50 import preprocess_input as resnet_preprocess
except Exception:
    import tensorflow as tf
    keras = tf.keras
    from tensorflow.keras.saving import register_keras_serializable
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# OpenCV (headless is fine)
import cv2

# ---------- Env / config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_resnet50.keras")
# If not provided, we’ll infer from model input
IMG_SIZE_ENV = os.getenv("IMG_SIZE")
USE_FACE = os.getenv("USE_FACE", "true").lower() not in {"0","false","no","off"}

CLASS_NAMES: List[str] = json.loads(os.getenv(
    "CLASS_NAMES", '["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]'
))

# ---------- Custom layer possibly embedded in the saved model ----------
@register_keras_serializable(package="faceit")
class ResNetPreprocess(keras.layers.Layer):
    def call(self, x):
        return resnet_preprocess(x)

# ---------- Model cache / metadata ----------
_model = None
_IMG_SIZE: Optional[int] = None
_HAS_PREPROCESS: Optional[bool] = None

def _infer_img_size(m) -> int:
    try:
        ishape = m.inputs[0].shape  # (None, H, W, C)
        h, w = int(ishape[1]), int(ishape[2])
        return h if h == w else max(h, w)
    except Exception:
        return 224

def _detect_embedded_preprocess(m) -> bool:
    # Look for our custom layer by name/type
    for l in m.layers:
        if l.__class__.__name__ == "ResNetPreprocess":
            return True
    return False

def get_model():
    """Load and cache the model and metadata."""
    global _model, _IMG_SIZE, _HAS_PREPROCESS
    if _model is None:
        _model = keras.models.load_model(
            MODEL_PATH, compile=False,
            custom_objects={"ResNetPreprocess": ResNetPreprocess},
        )
        _HAS_PREPROCESS = _detect_embedded_preprocess(_model)
        _IMG_SIZE = int(IMG_SIZE_ENV) if IMG_SIZE_ENV else _infer_img_size(_model)
    return _model

# ---------- OpenCV utilities (face detection & input normalization) ----------
# Pre-load cascade once
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _largest_face_bbox(gray: np.ndarray,
                       scaleFactor: float = 1.05,
                       minNeighbors: int = 3,
                       minSize: Tuple[int,int] = (20,20)
                      ) -> Optional[Tuple[int,int,int,int]]:
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=scaleFactor,
                                      minNeighbors=minNeighbors, minSize=minSize)
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
    x,y,w,h = faces[0]
    return int(x), int(y), int(w), int(h)

def _ensure_bgr(image: Union[np.ndarray, Image.Image, bytes, io.BytesIO, str]) -> np.ndarray:
    """
    Accepts: np.ndarray (assumed BGR), PIL.Image, bytes/BytesIO, or file path.
    Returns: OpenCV BGR np.ndarray with EXIF orientation handled.
    """
    # Already a cv2 image
    if isinstance(image, np.ndarray):
        return image

    # Bytes-like or PIL or path -> go via PIL to respect EXIF
    pil: Optional[Image.Image] = None
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))
    elif isinstance(image, io.BytesIO):
        pil = Image.open(image)
    elif isinstance(image, str):
        pil = Image.open(image)
    else:
        raise TypeError("Unsupported image type")

    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.array(pil)  # (H,W,3) RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def _prepare_tensor_from_bgr(bgr: np.ndarray, img_size: int, has_preprocess: bool) -> np.ndarray:
    """BGR -> RGB, resize, float32, optional external resnet_preprocess, add batch dim."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA).astype("float32")
    if not has_preprocess:
        rgb = resnet_preprocess(rgb)  # only if model doesn't include preprocessing
    x = np.expand_dims(rgb, 0)  # (1,H,W,3)
    return x

# ---------- Public API ----------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Backwards-compatible helper for the API that passes a PIL.Image.
    Does NOT do face detection (match previous behavior).
    """
    m = get_model()
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    pil_img = pil_img.resize((_IMG_SIZE, _IMG_SIZE), resample=Image.BILINEAR)
    arr = np.asarray(pil_img, dtype="float32")
    if not _HAS_PREPROCESS:
        arr = resnet_preprocess(arr)
    return np.expand_dims(arr, 0)

def predict_image(pil_img: Image.Image) -> dict:
    """
    Backwards-compatible: take a PIL image and return label/confidence/probabilities.
    No face cropping here (to keep the same API semantics).
    """
    model = get_model()
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }

def predict_any(image: Union[np.ndarray, Image.Image, bytes, io.BytesIO, str],
                topk: int = 3,
                use_face: Optional[bool] = None) -> Dict:
    """
    Teammate-style API: accepts path/PIL/bytes/BGR np.ndarray.
    Adds optional face detection & cropping, returns top3 and has_face.
    """
    model = get_model()
    img_size = _IMG_SIZE
    has_prep = _HAS_PREPROCESS
    use_face = USE_FACE if use_face is None else bool(use_face)

    bgr = _ensure_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    crop = None
    has_face = False
    if use_face:
        box = _largest_face_bbox(gray)
        if box is not None:
            x, y, w, h = box
            crop = bgr[y:y+h, x:x+w]
            has_face = True
    if crop is None:
        crop = bgr

    x = _prepare_tensor_from_bgr(crop, img_size, has_prep)
    probs = model.predict(x, verbose=0)[0]

    order = list(np.argsort(-probs))[:max(1, topk)]
    idx = int(order[0])
    top = [(CLASS_NAMES[i], float(probs[i])) for i in order]

    return {
        "label": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "top3": top,
        "raw": [float(v) for v in probs.tolist()],
        "has_face": has_face,
        # keep your existing shape too:
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }

__all__ = [
    "get_model",
    "preprocess_image",
    "predict_image",
    "predict_any",
    "ResNetPreprocess",
]
