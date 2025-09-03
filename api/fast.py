# TODO: Import your package, replace this by explicit imports of what you need
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faceit_package.model import get_model, predict_image
from PIL import Image
import io


app = FastAPI(title="FaceIt Emotions API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Please upload an image file")
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    return predict_image(img)


#@app.get("/predict")
#def get_predict(input_one: float,
#            input_two: float):
#    # TODO: Do something with your input
#    # i.e. feed it to your model.predict, and return the output
#    # For a dummy version, just return the sum of the two inputs and the original inputs
#    prediction = float(input_one) + float(input_two)
#    return {
#        'prediction': prediction,
#        'inputs': {
#            'input_one': input_one,
#            'input_two': input_two
#        }
#    }
