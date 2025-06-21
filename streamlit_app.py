import streamlit as st
import gdown
import tensorflow as tf
import os

@st.cache_resource
def load_model():
    model_path = "/tmp/inceptionv3_dr_v3_75acc.keras"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1vS8j1Ke0ZgUCrAGjHgt3eTXGEkw67epo"
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.title("DR Classifier")
st.success("Model loaded and ready!")
