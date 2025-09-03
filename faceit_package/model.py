from dotenv import load_dotenv
load_dotenv(override=True)  # re-read .env on each reload

from PIL import Image, ImageOps
import numpy as np
import os, json, tensorflow as tf




MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_model_FER_hannah.h5")
IMG_SIZE = int(os.getenv("IMG_SIZE", "48"))     # 48
CHANNELS = int(os.getenv("CHANNELS", "1"))      # 1
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_model_FER_hannah.h5")
CLASS_NAMES = json.loads(os.getenv(
    "CLASS_NAMES",
    '["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]'
))

# Parse CLASS_NAMES from .env (must be valid JSON, i.e., double quotes)
raw = os.getenv("CLASS_NAMES", '["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]')
try:
    CLASS_NAMES = json.loads(raw)
except Exception as e:
    raise ValueError(f"Invalid CLASS_NAMES (must be JSON list of strings): {raw!r}") from e



_model = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    # Match CRNO: grayscale → 48x48 → float32/255.0 → (1,48,48,1)
    pil_img = ImageOps.exif_transpose(pil_img)             # fix phone rotations
    pil_img = pil_img.convert("L")                          # grayscale
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    arr = np.array(pil_img, dtype="float32") / 255.0        # normalize
    arr = np.expand_dims(arr, axis=-1)                      # (48,48) -> (48,48,1)
    arr = np.expand_dims(arr, axis=0)                       # -> (1,48,48,1)
    return arr

def predict_image(pil_img: Image.Image) -> dict:
    model = get_model()
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]                  # assume softmax in last layer
    top = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[top],
        "confidence": float(probs[top]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
    }
