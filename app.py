import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image

# Page config
st.set_page_config(
    page_title="German Traffic Sign Recognition",
    layout="centered"
)

st.title("🚦 German Traffic Sign Recognition System")
st.write("Upload a traffic sign image to get the prediction")

# Load model
@st.cache_resource
def load_model():
    with open("model_comparison_results.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Load labels
labels = pd.read_csv("labels.csv")
class_names = labels["Name"].values

IMG_SIZE = 32

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = img / 255.0
    img = img.flatten()          # IMPORTANT for sklearn
    img = img.reshape(1, -1)     # (1, features)
    return img

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    img = preprocess_image(image)

    prediction = model.predict(img)
    class_id = int(prediction[0])

    # Confidence (if supported)
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(img)) * 100
        st.info(f"**Confidence:** {confidence:.2f}%")

    st.success(f"**Prediction:** {class_names[class_id]}")
