import streamlit as st
import torch
import cv2
import numpy as np
import os
import sqlite3
import pandas as pd
from transformers import BertTokenizer
from med3dinsight_main import MultiModalModel
from fpdf import FPDF
from datetime import datetime

# Initialize Database
conn = sqlite3.connect("hc_predictions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    patient_name TEXT,
    age INTEGER,
    scan_date TEXT,
    hc_value REAL,
    report_path TEXT
)
""")
conn.commit()

# Load trained model
model = MultiModalModel()
model.load_state_dict(torch.load("C:/med3dinsight/med3dinsight_main.pth", map_location=torch.device("cpu")))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to generate a unique Patient ID
def get_patient_id(name, age):
    cursor.execute("SELECT patient_id FROM predictions WHERE patient_name=? AND age=?", (name, age))
    existing_id = cursor.fetchone()
    
    if existing_id:  
        return existing_id[0]  # Return existing patient ID
    
    # Generate a new patient ID only if the patient does not exist
    new_id = f"PAT-{name[:3].upper()}-{age}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return new_id

# Function to predict HC
def predict_HC(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    text_report = "Fetal head measurement normal"
    text_tokens = tokenizer(text_report, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        hc_prediction = model(img, text_tokens["input_ids"], text_tokens["attention_mask"])
    
    return hc_prediction.item()

# Function to generate report PDF
def generate_pdf(image_bytes, image_filename, hc_value, patient_name, age, scan_date, patient_id):
    temp_image_path = f"C:/med3dinsight/temp_{image_filename}"
    with open(temp_image_path, "wb") as f:
        f.write(image_bytes)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Fetal Head Circumference Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(200, 10, f"Patient: {patient_name}", ln=True)
    pdf.cell(200, 10, f"Age: {age} years", ln=True)
    pdf.cell(200, 10, f"Scan Date: {scan_date}", ln=True)
    pdf.cell(200, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, f"Predicted HC: {hc_value:.2f} mm", ln=True)
    pdf.cell(200, 10, "Normal HC Range: 280 - 350 mm", ln=True)
    pdf.cell(200, 10, get_suggestions(hc_value), ln=True)
    pdf.ln(10)
    pdf.image(temp_image_path, x=50, y=120, w=100)
    
    pdf_output_path = f"C:/med3dinsight/{patient_name}_HC_Report.pdf"
    pdf.output(pdf_output_path)
    os.remove(temp_image_path)

    return pdf_output_path

# Function to provide suggestions based on HC Value
def get_suggestions(hc_value):
    if hc_value < 280:
        return "HC is lower than normal. Further examination needed. Possible causes: Growth restriction, genetic conditions."
    elif 280 <= hc_value <= 350:
        return "HC is within the normal range. No further action needed."
    else:
        return "HC is above the normal range. Consider additional tests. Possible causes: Hydrocephalus, excessive growth."

# Streamlit UI
st.set_page_config(page_title="Med3DInsight", page_icon="🩺", layout="centered")
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Predict HC", "View History"])

if option == "Predict HC":
    st.title("🧑‍⚕️ AI-Powered Fetal HC Prediction")

    with st.form("patient_details"):
        patient_name = st.text_input("👤 Patient Name")
        age = st.number_input("🎂 Age", min_value=0, max_value=100, step=1)
        scan_date = st.date_input("📅 Scan Date")
        uploaded_file = st.file_uploader("📤 Upload an Ultrasound Image", type=["png", "jpg", "jpeg"], key="file_uploader")
        submit = st.form_submit_button("🔍 Predict HC")

    if submit and uploaded_file:
        patient_id = get_patient_id(patient_name, age)
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="🖼️ Uploaded Ultrasound Image", use_column_width=True)
        hc_value = predict_HC(image_bytes)
        st.success(f"📏 Predicted HC: {hc_value:.2f} mm")
        st.info(get_suggestions(hc_value))
        
        pdf_path = generate_pdf(image_bytes, uploaded_file.name, hc_value, patient_name, age, scan_date, patient_id)
        
        cursor.execute("SELECT id FROM predictions WHERE patient_id=? AND scan_date=?", (patient_id, scan_date.strftime("%Y-%m-%d")))
        existing_record = cursor.fetchone()

        if not existing_record:  # Prevent duplicate scan date entries
            cursor.execute("""
                INSERT INTO predictions (patient_id, patient_name, age, scan_date, hc_value, report_path) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (patient_id, patient_name, age, scan_date.strftime("%Y-%m-%d"), hc_value, pdf_path))
            conn.commit()
        else:
            st.warning("⚠️ This patient record for the same scan date already exists!")

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(label="📥 Download Report", data=pdf_file, file_name=f"{patient_name}_HC_Report.pdf", mime="application/pdf")

elif option == "View History":
    st.title("📜 Previous Predictions")
    history = cursor.execute("SELECT id, patient_id, patient_name, age, scan_date, hc_value FROM predictions ORDER BY id DESC").fetchall()
    
    if history:
        df = pd.DataFrame(history, columns=["ID", "Patient ID", "Patient Name", "Age", "Scan Date", "HC Value (mm)"])
        st.dataframe(df)

        # Delete Option
        delete_id = st.number_input("Enter ID to Delete", min_value=1, step=1)
        if st.button("🗑️ Delete Record"):
            cursor.execute("DELETE FROM predictions WHERE id=?", (delete_id,))
            conn.commit()
            st.success("✅ Record Deleted! Refresh the page to see changes.")

conn.close()
