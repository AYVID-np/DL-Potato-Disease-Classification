# The image classification service is built using FastAPI and integrates with TensorFlow Serving to classify images into different 
# categories. The service exposes two endpoints: a ping endpoint for testing connectivity and a prediction endpoint for making image 
# classification requests using a pre-trained TensorFlow model.

from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8509/v1/models/potato_classification_model:predict"

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

    json_data = {
        "instances" : image_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    print("response", response.json())
    prediction = np.array(response.json()["predictions"][0])
        
    pred_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "predicted class": pred_class,
        "Confidence" : float(confidence)
    }
    


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=1111)