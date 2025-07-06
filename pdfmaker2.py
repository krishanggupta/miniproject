# %% [markdown]
# # Library import section

# %%
import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
from datetime import datetime

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
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from PyPDF2 import PdfMerger

# %% [markdown]
# # PDF Content Code

# %%
def merge_pdfs(pdf_paths, output_path):
    merger = PdfMerger()
    for path in pdf_paths:
        merger.append(path)
    merger.write(output_path)
    merger.close()
    return output_path

def add_disclaimer(canvas_obj, text):
    canvas_obj.setFont("Times-Italic", 6)
    canvas_obj.setFillColor(colors.grey)
    canvas_obj.drawString(20 * mm, 15 * mm, text)
    canvas_obj.setFillColor(colors.black)  # Reset color

def draw_separator(c, y_pos):
    c.setStrokeColor(green)
    c.setLineWidth(2)
    c.line(20 * mm, y_pos, 190 * mm, y_pos)


def draw_bold_underline(c, text, x, y, font_size=12, line_color=black,type='None'):
    c.setFont("Times-Bold", font_size)
    if type=='right':
        c.drawRightString(x,y,text)
        
    else:
        c.drawString(x, y, text)
   
    # Measure text width and draw underline manually
    text_width = c.stringWidth(text, "Times-Bold", font_size)
    underline_y = y - 2  # 2 points below the baseline

    c.setStrokeColor(line_color)
    c.setLineWidth(1)
    if type=='right':
        c.line(x-text_width, underline_y, x, underline_y)
        
    else:
        c.line(x, underline_y, x + text_width, underline_y)


def draw_table_on_canvas(c,x,y,data):

    table = Table(data, colWidths=[90 * mm, 80* mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))

    # Wrap and draw on canvas
    x = x * mm
    y = y * mm
    table.wrapOn(c, 0, 0)
    table.drawOn(c, x, y)


def section1(canvas_obj):
    # Left Company Logo
    logo_path = "eyelogopng.png"  # replace with your logo path
    canvas_obj.drawImage(logo_path, 20 * mm, 267 * mm, width=30 * mm, height=30 * mm)

    # Right VNIT EYECARE text
    canvas_obj.setFont("Times-Bold", 18)
    canvas_obj.drawRightString(190 * mm, 285 * mm, "VNIT EYECARE")

    # Separator line
    draw_separator(canvas_obj, 265 * mm)
    

def section2(canvas_obj,**params2):
    try:
        patient_info = params2.get('patient_info')
        general_info = params2.get('general_info')
    except:
        pass

    draw_bold_underline(canvas_obj,'Patient Details',20*mm,255*mm)

    # Left Column: Patient Info
    if patient_info==None:
        patient_info = {
            "Name": "John Doe",
            "Patient ID": "123456",
            "DOB": "1990-01-01",
            "Gender": "Male"
        }

    y = 249 * mm
    for key, value in patient_info.items():
        canvas_obj.drawString(20 * mm, y, f"{key}: {value}")
        y -= 6 * mm

    # Right Column: General Info
    draw_bold_underline(canvas_obj,'General Information',190*mm,255*mm,type='right')
    if general_info==None:
        general_info = {
        "Ref. Doctor": "Dr. Smith",
        "Test Date-Time": "2025-07-06 14:30",
        "Report Date-Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Performed By": "Technician A"
    }

    y = 249 * mm
    for key, value in general_info.items():
        canvas_obj.drawRightString(190 * mm, y, f"{key}: {value}")
        y -= 6 * mm

    # Separator line
    draw_separator(canvas_obj, y - 2 * mm)

def section3(canvas_obj):
   
    # Write AI Generated Report
    canvas_obj.setFont("Times-Bold", 18)
    canvas_obj.drawString(20 * mm, 210 * mm, "   AI GENERATED DIABETIC RETINOPATHY REPORT   ")

    # Separator line
    draw_separator(canvas_obj, 200 * mm)

def section4(canvas_obj,image_path,**params):
    # Put the captured image in the centre of the page
    # Separator line
    canvas_obj.drawImage(image_path, 50 * mm, 90 * mm, width=100 * mm, height=100 * mm)
    draw_separator(canvas_obj, 80 * mm)

def section5(canvas_obj,prob_data):
    draw_bold_underline(canvas_obj,'Findings',20*mm,70*mm)

    data = [
        ['Stage', 'Probability (%)'],
        ['A', '10'],
        ['B', '10'],
        ['C', '20'],
        ['D','60']
    ]
    if prob_data!=None:
        for i in range(len(prob_data)):
            data[i+1][0]=prob_data[i][0]
            data[i+1][1]=str(prob_data[i][0])
            

    draw_table_on_canvas(canvas_obj,20,35,data)

def section6(canvas_obj, general_prediction):
    draw_bold_underline(canvas_obj,'Clinical Significance',20*mm,255*mm)
    canvas_obj.drawString(text=general_prediction,x=20*mm, y=235*mm)
    draw_separator(canvas_obj,235*mm)


def create_pdf(filename,prob_data,patient_info,general_info):
    # patient_info = {
    #         "Name": "John Doe",
    #         "Patient ID": "123456",
    #         "DOB": "1990-01-01",
    #         "Gender": "Male"
    #     }
    # general_info = {
    #     "Ref. Doctor": "Dr. Smith",
    #     "Test Date-Time": "2025-07-06 14:30",
    #     "Report Date-Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    #     "Performed By": "Technician A"
    # }

    c = canvas.Canvas(filename, pagesize=A4)
    section1(c)
    section2(c,**{'patient_info':patient_info,'general_info':general_info})
    section3(c)
    section4(c,'fn.jpg')
    section5(c,prob_data)
    add_disclaimer(c,
                   text="Disclaimer: This report is AI-generated and for reference purpose only. Please consult with an Ophthalmologist for a comprehensive evaluation and personalized treatment plan.")
    #c.showPage()
    #section1(c)
    #section6(c,'randomtext')
    c.save()


def create_pdf_helper(report_text):
    # === Local re-use of section1 and disclaimer ===
    def section1_on_canvas(canvas_obj):
        logo_path = "eyelogopng.png"
        canvas_obj.drawImage(logo_path, 20 * mm, 267 * mm, width=30 * mm, height=30 * mm)
        canvas_obj.setFont("Times-Bold", 18)
        canvas_obj.drawRightString(190 * mm, 285 * mm, "VNIT EYECARE")
        draw_separator(canvas_obj, 265 * mm)


    def add_disclaimer(canvas_obj, text):
        canvas_obj.setFont("Times-Italic", 6)
        canvas_obj.setFillColor(grey)
        canvas_obj.drawString(20 * mm, 15 * mm, text)
        canvas_obj.setFillColor("black")  # Reset color

    # === Header/Footer callback for Platypus ===
    def add_section1_and_disclaimer(canvas_obj, doc):
        canvas_obj.saveState()

        # Section1 header
        section1_on_canvas(canvas_obj)

        # # Optional: watermark
        # canvas_obj.translate(A4[0] / 2, A4[1] / 2)
        # canvas_obj.rotate(45)
        # canvas_obj.setFont('Helvetica-Bold', 36)
        # canvas_obj.setFillColor(grey, alpha=0.2)
        # canvas_obj.drawCentredString(0, 0, "Generated by VNIT MedAssistant AI")
        # canvas_obj.restoreState()

        # Disclaimer
        canvas_obj.saveState()
        add_disclaimer(canvas_obj, text="Disclaimer: This report is AI-generated and for reference purpose only. Please consult with an Ophthalmologist for a comprehensive evaluation and personalized treatment plan.")
        canvas_obj.restoreState()

    # === Platypus PDF build ===
    pdf_path = "description.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=100, bottomMargin=72)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=14, leading=18, spaceAfter=12, spaceBefore=12, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="NormalJustified", alignment=TA_JUSTIFY, fontSize=11, leading=16))

    flowables = []

    for paragraph in report_text.strip().split("\n\n"):
        if paragraph.strip().startswith("**") and "**" in paragraph.strip()[2:]:
            header_text = paragraph.strip().replace("**", "")
            flowables.append(Paragraph(header_text, styles["SectionHeader"]))
        else:
            flowables.append(Paragraph(paragraph.replace("\n", "<br/>"), styles["NormalJustified"]))
        flowables.append(Spacer(1, 8))

    doc.build(flowables, onFirstPage=add_section1_and_disclaimer, onLaterPages=add_section1_and_disclaimer)

    return pdf_path



if __name__=='__main__':
    create_pdf("custom_report.pdf",prob_data=None)
    # Step 2: Generate text-based Platypus PDF
    desc_path = create_pdf_helper(
        report_text="""
        The patient is showing signs of moderate non-proliferative diabetic retinopathy. The patient is showing signs of moderate non-proliferative diabetic retinopathy

        **Recommendation**

        Refer to a retina specialist for further diagnosis.
        """)

    # Step 3: Merge both PDFs
    merge_pdfs(["custom_report.pdf", desc_path], "final_report.pdf")
    print("✅ Merged PDF saved as final_report.pdf")
    



