import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

import google.generativeai as genai
from fpdf import FPDF
import tempfile
import os

# ---- Page Configuration ----
st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="wide")



# Initialize Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel(model_name="models/gemini-pro")


def generate_report_text(predicted_class, confidence, stage_prob,p_name='Krishang',
                         p_age=23,p_gender='Male'):
    prompt = f"""
    Create a short and personalized diagnostic report for a diabetic retinopathy screening.
    The predicted stage is: {predicted_class}, with a model confidence of {confidence:.2f}%. The results
    of predicted class might be slightly inaccurate hence, base your report on the probabilities that
    were predicted for different stages: {stage_prob}. 

    Your report should be based on personalised details i.e suggestions based on age given by {p_age}, 
    gender given by {p_gender}. Mention the person's name in the report given by {p_name}.

    Explain the implications of this stage to a non-medical person, suggest next medical steps,
    and emphasize the importance of regular eye exams. Make it clear, compassionate, and supportive.
    """
    response = model.generate_content(prompt)
    return response.text

def create_pdf(report_text, predicted_class):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Diabetic Retinopathy Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"\nDiagnosis: {predicted_class}\n")
    pdf.multi_cell(0, 10, report_text)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name



# ---- Load ONNX model ----
@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession("mlmodel.onnx")  # Must be in same folder or correct relative path
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    return session, input_name, input_shape

session, input_name, input_shape = load_onnx_model()

# ---- Class Labels ----
class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# ---- Sidebar ----
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📷 Classify Image", "📊 Model Info", "📚 DR Stages", "👨‍⚕️ About Me"])

# ---- Tab 1: Classify Image ----
if page == "📷 Classify Image":
    st.title("🩺 Diabetic Retinopathy Classifier")
    st.write("Upload a retina (fundus) image below and let the model predict the DR stage.")

    uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        target_size = tuple(input_shape[1:3])  # Assumes (None, H, W, C)
        image = image.resize(target_size)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.write(f"🔍 Input shape for ONNX: {img_array.shape}")
        st.write(f"🧠 Model expects shape: {input_shape}")

        # Predict
        try:
            prediction = session.run(None, {input_name: img_array})[0][0]
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
            st.markdown(f"### 📈 Confidence: `{confidence:.2f}%`")
            st.markdown("#### 🔍 Full Prediction Probabilities:")
            st.bar_chart({label: float(prob) for label, prob in zip(class_labels, prediction)})
            stage_prob={label: float(prob) for label, prob in zip(class_labels, prediction)}
            if st.button("📝 Generate Custom Report PDF"):
                with st.spinner("Generating report..."):
                    report_text = generate_report_text(predicted_class, confidence, stage_prob)
                    pdf_path = create_pdf(report_text, predicted_class)

                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 Download PDF Report", f, file_name="DR_Report.pdf", mime="application/pdf")

                os.remove(pdf_path)

        except Exception as e:
            st.error(f"❌ Model inference failed: {str(e)}")


# ---- Tab 2: Model Info ----
elif page == "📊 Model Info":
    st.title("📊 Model Information")
    st.markdown("""
    - **Model Type**: DenseNet121 (converted to ONNX)
    - **Trained On**: Labeled Indian DR Dataset
    - **Input Size**: 224x224 RGB (normalized float32)
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

# ---- Tab 4: About Me ----
elif page == "👨‍⚕️ About Me":
    st.title("👨‍💻 About This Project")
    st.markdown("""
    This app was created by **Krishang Gupta** as part of a machine learning project to automate the classification of Diabetic Retinopathy using deep learning.

    - 🔬 Based on Densenet121, converted to ONNX
    - 🧠 Inference done using ONNX Runtime (faster + cloud-ready)
    - 🧑‍🏫 Goal: Assist medical professionals in early detection of DR

    [📧 Contact](mailto:krishanggupta.kg@gmail.com) | [🌐 LinkedIn](https://www.linkedin.com/in/krishang-gupta-0b0041305/)
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/0/06/Retinopathy3.jpg", caption="Sample DR Image")
