from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from faceit_package.model import get_model, predict_image, predict_any
from PIL import Image
import io

app = FastAPI(title="FaceIt Emotions API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 5 * 1024 * 1024  # 5 MB max upload


@app.on_event("startup")
def warmup():
    # Load model once at startup
    get_model()


@app.get("/")
def root():
    return {"message": "FaceIt API up"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file (jpg/png)"),
    use_face: bool = Query(
        True, description="Detect largest face and crop before inference"
    ),
    topk: int = Query(
        3, ge=1, description="Number of top classes to include in 'top3' (or topk)"
    ),
):
    """
    Teammate-style endpoint:
    - accepts image upload
    - optionally crops to largest detected face
    - returns label, confidence, topk, raw probabilities, has_face, probabilities
    """
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png).")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_BYTES // (1024*1024)} MB).")

    try:
        # predict_any can take bytes directly
        result = predict_any(data, topk=topk, use_face=use_face)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/predict-simple")
async def predict_simple(
    file: UploadFile = File(..., description="Image file (jpg/png)")
):
    """
    Backwards-compatible endpoint:
    - no face detection/cropping
    - returns label, confidence, probabilities
    """
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png).")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_BYTES // (1024*1024)} MB).")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        return predict_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
