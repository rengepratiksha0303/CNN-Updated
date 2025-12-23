from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Load class labels
labels = pd.read_csv("labels.csv")
class_names = labels["Name"].values

IMG_SIZE = 32

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = os.path.join("static", file.filename)
            file.save(image_path)

            img = preprocess_image(image_path)
            preds = model.predict(img)
            class_id = np.argmax(preds)
            prediction = class_names[class_id]
            confidence = round(float(np.max(preds)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
