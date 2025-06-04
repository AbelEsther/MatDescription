import streamlit as st
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import gdown
import zipfile

# --------------------------------------
# STEP 1: Download and extract model
# --------------------------------------

# Only download if model folder doesn't exist
model_dir = "train_model"
zip_file = "train_model.zip"
drive_file_id ="1-2dR71dBKp4mQgId4uNLI9uVvEdnFYQ-" # <-- Replace this with your real file ID

if not os.path.exists(model_dir):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id="1-2dR71dBKp4mQgId4uNLI9uVvEdnFYQ-", zip_file, quiet=False)

    st.info("Extracting model files...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(model_dir)

    st.success("Model loaded!")

# --------------------------------------
# STEP 2: Load model
# --------------------------------------

model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

# --------------------------------------
# STEP 3: Define UI and prediction logic
# --------------------------------------

st.title("🛠️ Material Name Generator")
st.write("Upload a file with `System_1` descriptions. The model will generate standardized short names.")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

def generate_short_name(text, max_length=40):
    if not isinstance(text, str):
        return ""
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Check column
    if 'System_1' not in df.columns:
        st.error("Your file must have a 'System_1' column.")
    else:
        with st.spinner("Generating short names..."):
            df["Predicted_System_2"] = df["System_1"].apply(generate_short_name)

        st.success("✅ Prediction complete!")
        st.dataframe(df.head())

        # Download
        output_path = "predicted_output.xlsx"
        df.to_excel(output_path, index=False)

        with open(output_path, "rb") as f:
            st.download_button("📥 Download Result", f, file_name="predicted_output.xlsx")
