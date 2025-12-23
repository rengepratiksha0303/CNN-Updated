import streamlit as st
import numpy as np
import pickle
from PIL import Image

st.set_page_config(page_title="German Traffic Sign Recognition")

st.title("🚦 German Traffic Sign Recognition System")

# Load model
@st.cache_resource
def load_model():
    with open("model_comparison_results.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# GTSRB class names (OFFICIAL)
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield",
    "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

IMG_SIZE = 32

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = img.flatten().reshape(1, -1)
    return img

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    img = preprocess_image(image)
    prediction = model.predict(img)
    class_id = int(prediction[0])

    st.success(f"**Prediction:** {CLASS_NAMES[class_id]}")

    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(img)) * 100
        st.info(f"**Confidence:** {confidence:.2f}%")
