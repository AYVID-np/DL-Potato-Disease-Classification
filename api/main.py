# The image classification service is built using FastAPI and leverages a pre-trained TensorFlow model for classifying images into 
# different categories. The service exposes two endpoints: a ping endpoint for testing connectivity and a prediction endpoint for making image classification requests. 
# CORS is enabled to allow requests from any origin

from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app = FastAPI()

# Allow all origins for demonstration purposes (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("C:/Users/91987/Documents/DLProjects/saved_models/1")
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return "Hello"

def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    # Input: [256, 256,3] But predict takes [[256,256, 3]]
    image_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(image_batch) 
    
    pred_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    print({
        "predicted class": pred_class,
        "Confidence" : float(confidence)
    })

    return {
        "predicted class": pred_class,
        "Confidence" : float(confidence)
    }
    


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=9090)