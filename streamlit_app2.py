import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ---- Page Configuration ----
st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="wide")

# ---- Load model ----
@st.cache_resource
def load_trained_model():
    return load_model("/Users/krishanggupta/Desktop/MyFiles/college/Sem2/Mini Project/Dataset/DR Dataset_merged/Models/Evaluation2Modesl/densenet121_dr_finetuned_73acc.h5")  # your model path

model = load_trained_model()

# ---- Class Labels ----
class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# ---- Sidebar ----
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📷 Classify Image", "📊 Model Info", "📚 DR Stages", "👨‍⚕️ About Us"])

# ---- Tab 1: Classify Image ----
if page == "📷 Classify Image":
    st.title("🩺 Diabetic Retinopathy Classifier")
    st.write("Upload a retina (fundus) image below and let the model predict the DR stage.")

    uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        image = image.resize((300, 300))
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(image_array)[0]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
        st.markdown(f"### 📈 Confidence: `{confidence:.2f}%`")

        st.markdown("#### 🔍 Full Prediction Probabilities:")
        st.bar_chart({label: float(prob) for label, prob in zip(class_labels, prediction)})

# ---- Tab 2: Model Info ----
elif page == "📊 Model Info":
    st.title("📊 Model Information")
    st.markdown("""
    - **Model Type**: EfficientNetB3
    - **Trained On**: Labeled Indian DR Dataset
    - **Input Size**: 300x300 RGB
    - **Classes**: No DR, Mild, Moderate, Severe, Proliferative DR
    - **Accuracy Achieved**: ~90% (fine-tuned with class weights)
    - **Loss Function**: Categorical Crossentropy
    - **Optimizer**: Adam
    """)

    st.image("https://miro.medium.com/v2/resize:fit:1400/1*nzO4e6pgvL3uq1Ap46adkA.png", caption="EfficientNet Architecture (source: Medium)")

# ---- Tab 3: DR Stages ----
elif page == "📚 DR Stages":
    st.title("📚 Diabetic Retinopathy Stages")

    stages = {
        "No DR": "No visible damage to the retina.",
        "Mild": "Microaneurysms begin to appear.",
        "Moderate": "Blood vessels in the retina are blocked.",
        "Severe": "Many more blood vessels are blocked, retina is damaged.",
        "Proliferative DR": "New abnormal blood vessels grow; risk of vision loss is high."
    }

    for stage, desc in stages.items():
        st.markdown(f"### {stage}")
        st.write(desc)
        st.progress((list(stages.keys()).index(stage) + 1) / len(stages))

# ---- Tab 4: About Us ----
elif page == "👨‍⚕️ About Us":
    st.title("👨‍💻 About This Project")
    st.markdown("""
    This app was created by **Krishang Gupta** as part of a machine learning project to automate the classification of Diabetic Retinopathy using deep learning.

    - 🔬 Based on EfficientNetB3, fine-tuned on real Indian DR images
    - 🧠 Model trained with data augmentation & class balancing
    - 🧑‍🏫 Goal: Assist medical professionals in early detection of DR

    [📧 Contact](mailto:krishang@example.com) | [🌐 LinkedIn](https://linkedin.com/in/krishanggupta)
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/0/06/Retinopathy3.jpg", caption="Sample DR Image")

