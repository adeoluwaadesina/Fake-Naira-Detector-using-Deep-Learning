import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the Keras model
model = tf.keras.models.load_model('naira_model.h5')

# Define function to preprocess image and make predictions
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((256, 256))  # Resize to match the model's input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image.astype(np.float32)  # Convert to FLOAT32
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

# Streamlit app with custom CSS
st.set_page_config(page_title="Nigeria Currency Classifier", page_icon="💵")

st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333333;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2em;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-size: 1em;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .prediction-text {
        font-size: 1.5em;
        margin-top: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💵 Nigeria Currency Classifier")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Clear previous prediction when new image is uploaded
    st.empty()

    if st.button("Predict"):
        prediction = predict(image)
        if prediction[0][0] > prediction[0][1]:  # Assuming index 0 corresponds to "fake" and index 1 corresponds to "real"
            st.markdown('<p class="prediction-text">Prediction: Fake Naira Note</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="prediction-text">Prediction: Real Naira Note</p>', unsafe_allow_html=True)
