import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# ---- Page Configuration ----
st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="wide")

# ---- Load ONNX model ----
@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession("mlmodel.onnx")
    return session, session.get_inputs()[0].name

session, input_name = load_onnx_model()

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
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Predict
        prediction = session.run(None, {input_name: img_array})[0][0]
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
    - **Model Type**: Densenet121 (converted to ONNX)
    - **Trained On**: Labeled Indian DR Dataset
    - **Input Size**: 300x300 RGB
    - **Classes**: No DR, Mild, Moderate, Severe, Proliferative DR
    - **Accuracy Achieved**: ~73%
    - **Inference Engine**: ONNX Runtime (no TensorFlow required!)
    """)

    st.image("https://miro.medium.com/v2/resize:fit:1400/1*ckLNL5fx3JNhgNzKeOnx_w.png", caption="Densenet Architecture")

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

    - 🔬 Based on Densenet121, converted to ONNX
    - 🧠 Inference done using ONNX Runtime (faster + cloud-ready)
    - 🧑‍🏫 Goal: Assist medical professionals in early detection of DR

    [📧 Contact](mailto:krishang@example.com) | [🌐 LinkedIn](https://linkedin.com/in/krishanggupta)
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/0/06/Retinopathy3.jpg", caption="Sample DR Image")
