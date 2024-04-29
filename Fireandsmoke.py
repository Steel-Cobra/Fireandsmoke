import os, cv2
import numpy as np
import random
import keras
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify
import json
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException, UploadFile, File

app = FastAPI()

# model = pickle.load(open('model.pkl','rb'))
from keras.models import load_model

# Load the model
fire_smoke = load_model('C:/Users/abdul/Downloads/fire_smoke.h5')


@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()

    # Convert image data to numpy array
    nparr = np.frombuffer(contents, np.uint8)

    # Decode image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform any image processing here, e.g., resizing
    resized_image = cv2.resize(image, (128, 128))
    image_exp = np.expand_dims(resized_image, axis=0)
    preds = fire_smoke.predict(image_exp)[0]
    # # processed = preprocess(image.file)
    # # preds = model.predict_proba(processed)[0]
    # preds = 1
    print(preds)
    # print(type(image))
    if preds > 0.7:
        return {"message": "fire"}
    return {"message": "smoke"}


# @app.get('/{text}')
# async def hello_world(text):
#     return {'message':f"User Input Text is : {text}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)