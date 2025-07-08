import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
from datetime import datetime
import re

import google.generativeai as genai
from fpdf import FPDF
import tempfile
import os
from datetime import date

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import grey

from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import green, black
from reportlab.lib.units import mm
from reportlab.lib.units import inch

from reportlab.lib.colors import black  # or import other colors like red, blue, etc.

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import pdfmaker2




st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="wide")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
#genai.configure(api_key="zaSyAIhrSVwoKg69NrmWhE_e-B34zkWz5nZJ8")

model = genai.GenerativeModel(model_name="gemini-2.0-flash")

if "user_info" not in st.session_state:
    st.session_state.user_info = {
        "name": "Patient A",
        "age": None,
        "gender": "Unspecified",
        "dob": None,
        "image_path": None
    }
def generate_report_text(predicted_class, confidence, stage_prob):
    
    info = st.session_state.user_info
    prompt = f"""
    You are generating a medical report. Start directly. 
   
    You are an AI Assitant called, VNIT MedAssistant! Introduce yourself first and greet the user whose name is
    {info['name']}. Give introduction by Keeping in mind , Age: {info['age']}, Gender: {info['gender']} of user and address accordingly.
    
    Do not include any conversational phrases such as “Okay”, “Sure”, “Here's the response”. 
    Write as if this is the final version being printed in a formal medical document.
    Always start printing with 'Greetings! I am VNIT MedAssistant! How are you doing? After this, greet the user with
    the name 
    Dont use any other approach to start. Never mention date or anything else. Just keep in mind user's age and
    user's gender while you address him/her.


    You are to explain the implications of the The predicted diabteic retinopathy stage which is: {predicted_class} in one
    section, suggest next medical steps in another section
    and emphasize the importance of regular eye exams. 
    Make it clear, compassionate, and supportive.
    We have created a short and personalized diagnostic report for a diabetic retinopathy screening.
    Dont give any answer in bullets. Make it paragraph. If bullet is required then instead separate it via one line


    At the end, say thank you.   

    """
    response = model.generate_content(prompt)
    return response.text

def generate_report_text2(predicted_class, confidence, stage_prob):
    info = st.session_state.user_info
    greeting_prompt = f"""
    You are VNIT MedAssistant! Introduce yourself first. Greet the person name as per:
    Patient Name: {info['name']}
    Format: 
    Hello "patient name", I am VNIT MedAssistant, an AI designed to help provide preliminary assessments
    based on available data. I'm here to assist in understanding your diabetic retinopathy
    screening results.
    """

    intro_prompt=f"""
    Create a short and personalized diagnostic report for a diabetic retinopathy screening.
    The predicted diabetic retinopathy stage is: {predicted_class}, with a model confidence of {confidence:.2f}%.
    The probabilities of prediction are {stage_prob}. Explain the results focusing mainly on max chances
    based on Base your report on the probabilities for different stages: {stage_prob}.
    Content is under heading:
    "Understanding the Results:" 
    Example :
    {info['name']}, this means that there are some noticeable changes in the blood vessels of your
    retina due to diabetes. "Moderate" diabetic retinopathy implies that there's more than just
    mild damage, but it's not yet at the most advanced stages. You may have some blocked
    blood vessels, which can affect how well your retina is working. While your vision might not
    be significantly affected yet, it's crucial to take action now to prevent it from getting worse.
    """
    next_steps_prompt=f"""
    Explain the implications of {predicted_class}, suggest next medical steps,
    and emphasize the importance of regular eye exams. Make it clear, compassionate, and supportive.
    Explain it under 

    """
    response = model.generate_content(greeting_prompt)
    return response.text

def add_watermark(canvas, doc):
    canvas.saveState()
    canvas.translate(A4[0] / 2, A4[1] / 2)
    canvas.rotate(45)
    canvas.setFont('Helvetica-Bold', 36)
    canvas.setFillColor(grey, alpha=0.2)
    canvas.drawCentredString(0, 0, "Generated by VNIT MedAssistant AI")
    canvas.restoreState()

    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(grey)
    footer_text = "Disclaimer: This report is AI-generated and for reference purpose only. Please consult with an Ophthalmologist for a comprehensive evaluation and personalized treatment plan."
    canvas.drawCentredString(A4[0] / 2, 0.5 * inch, footer_text)
    canvas.restoreState()


@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession("mlmodel.onnx")
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    return session, input_name, input_shape

session, input_name, input_shape = load_onnx_model()
class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

tab1, tab2, tab3, tab4, tab5 = st.tabs(["👤 User Info", "📷 Classify Image",  "📚 DR Stages", "📊 Model Info", "👨‍⚕️ About Me"])
with tab1:
    st.header("👤 Enter User Information")
    
    # Patient Information
    st.subheader("🧍 Patient Details")
    st.session_state.user_info["name"] = st.text_input("Full Name")
    st.session_state.user_info["gender"] = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
    
    age_option = st.radio("Provide Age Info", ["Date of Birth", "Age"])
    if age_option == "Date of Birth":
        dob = st.date_input("Date of Birth", value=date.today(), min_value=date(1900, 1, 1), max_value=date.today())
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        st.session_state.user_info["age"] = f"{age}yrs"
        st.session_state.user_info["dob"] = dob.strftime("%Y-%m-%d")
    else:
        age = st.number_input("Age (in years)", min_value=1, max_value=120)
        st.session_state.user_info["age"] = f"{int(age)}yrs"
        st.session_state.user_info["dob"] = "N/A"

    st.session_state.user_info["patient_id"] = st.text_input("Patient ID")

    # General Information
    st.subheader("📋 General Details")
    st.session_state.general_info = {}
    st.session_state.general_info["ref_doctor"] = st.text_input("Referring Doctor")
    st.session_state.general_info["test_datetime"] = st.text_input("Test Date-Time", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
    st.session_state.general_info["report_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.general_info["performed_by"] = st.text_input("Performed By", value="Technician A")



with tab2:
    st.header("📷 Upload Fundus Image")
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])
    patient_info = {
    "Name": st.session_state.user_info.get("name", ""),
    "Patient ID": st.session_state.user_info.get("patient_id", ""),
    "Age":  st.session_state.user_info.get("age",f"{age}yrs"),
"Gender": st.session_state.user_info.get("gender", "")
     }

    general_info = {
    "Ref. Doctor": st.session_state.general_info.get("ref_doctor", ""),
    "Test Date-Time": st.session_state.general_info.get("test_datetime", ""),
    "Report Date-Time": st.session_state.general_info.get("report_datetime", datetime.now().strftime("%Y-%m-%d %H:%M")),
    "Performed By": st.session_state.general_info.get("performed_by", "")
    }
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        target_size = tuple(input_shape[1:3])
        image = image.resize(target_size)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        
        prediction = session.run(None, {input_name: img_array})[0][0]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        stage_prob = {label: float(prob) for label, prob in zip(class_labels, prediction)}

        st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
        st.markdown(f"### 📈 Confidence: `{confidence:.2f}%`")
        st.markdown("#### 🔍 Full Prediction Probabilities:")
        st.bar_chart(stage_prob)

    if st.button("📝 Generate Report PDF"):
        with st.spinner("Generating report..."):
            temp_image_path = "temp_fundus.png"
            image.save(temp_image_path)

            # Delete old PDFs if they exist
            for file in ['DRprediction.pdf', 'description.pdf']:
                if os.path.exists(file):
                    os.remove(file)

            # Get fresh report content
            report_text = generate_report_text(predicted_class, confidence, stage_prob)

            # Generate both PDFs
            pdf1 = pdfmaker2.create_pdf('DRprediction.pdf', stage_prob, patient_info, general_info, temp_image_path)
            pdf2_path = pdfmaker2.create_pdf_helper(report_text)

            # Merge into final report
            merged_pdf_path = pdfmaker2.merge_pdfs(['DRprediction.pdf', pdf2_path],'merged.pdf')

            # Serve to user
            pn=st.session_state.user_info.get("name", "")
            pid=st.session_state.user_info.get("patient_id", "")

            with open(merged_pdf_path, "rb") as f:
                    st.download_button("📥 Download PDF Report", f, file_name=pn+'_'+pid+'.pdf', mime="application/pdf")

            os.remove(merged_pdf_path)
            os.remove(temp_image_path)


with tab3:
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

with tab4:
    st.title("📊 Model Information")
    st.markdown("""
    - **Model Type**: InceptionV3
    - **Trained On**: APTOS 2019 Dataset
    - **Input Size**: 224x224 RGB (normalized float32)
    - **Classes**: No DR, Mild, Moderate, Severe, Proliferative DR
    - **Accuracy Achieved**: ~73%
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*ckLNL5fx3JNhgNzKeOnx_w.png", caption="Densenet Architecture")


with tab5:
    st.title("👨‍💻 About This Project")
    st.markdown("""
    This app was created by **Krishang Gupta** as part of a machine learning project to automate the classification of Diabetic Retinopathy using deep learning.

    - 🔬 Based on CNN Architecture
    - 🧠 Inference done using ONNX Runtime 
    - 🧑‍🏫 Goal: Assist medical professionals in early detection of DR.

    [📧 Contact](mailto:krishanggupta.kg@gmail.com) | [🌐 LinkedIn](https://www.linkedin.com/in/krishang-gupta-0b0041305/)
    """)
    
