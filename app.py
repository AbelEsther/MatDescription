full_app_with_features = '''
import streamlit as st
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os

# --------------------------------------
# PAGE CONFIGURATION
# --------------------------------------
st.set_page_config(
    page_title="Material Name Generator + Deduplication",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------
# LOAD MODEL (download from Google Drive if needed)
# --------------------------------------
import zipfile
import gdown

@st.cache_resource
def download_and_load_model():
    model_dir = "train_model"
    model_zip = "train_model.zip"
    file_id = "1-2dR71dBKp4mQgId4uNLI9uVvEdnFYQ-"  # Your actual file ID
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_dir):
        with st.spinner("⏬ Downloading model from Google Drive..."):
            gdown.download(url, model_zip, quiet=False)
            with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                zip_ref.extractall(model_dir)

    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = download_and_load_model()

# --------------------------------------
# PAGE HEADER
# --------------------------------------
st.title("🧠 Material Name Generator & Deduplicator")
st.markdown("This tool standardizes material descriptions and optionally checks for duplicates against your ERP system.")
st.markdown("---")

# --------------------------------------
# FILE UPLOAD
# --------------------------------------
st.header("📥 Upload New Material File")
uploaded_file = st.file_uploader("Upload CSV or Excel file with `System_1` column", type=["csv", "xlsx"])

def generate_short_name(text, max_length=40):
    if not isinstance(text, str):
        return ""
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    if 'System_1' not in df.columns:
        st.error("The uploaded file must have a 'System_1' column.")
    else:
        with st.spinner("🔄 Generating standardized names..."):
            df["Predicted_System_2"] = df["System_1"].apply(generate_short_name)

        st.success("✅ Standardization complete!")
        st.markdown("### 📊 Preview of Standardized Results")
        st.dataframe(df.head(10))

        # Ask if user wants to deduplicate
        if st.checkbox("🔍 Check for duplicates against ERP data"):
            erp_file = st.file_uploader("Upload ERP Material List (CSV or Excel)", type=["csv", "xlsx"], key="erp_file")

            if erp_file:
                df_erp = pd.read_csv(erp_file) if erp_file.name.endswith(".csv") else pd.read_excel(erp_file)

                if "Standard_Name" not in df_erp.columns:
                    st.error("ERP file must contain a 'Standard_Name' column.")
                else:
                    df["normalized_predicted"] = df["Predicted_System_2"].str.strip().str.lower()
                    existing_names = df_erp["Standard_Name"].astype(str).str.strip().str.lower().unique()
                    df["Is_Duplicate"] = df["normalized_predicted"].isin(existing_names)
                    df["Status"] = df["Is_Duplicate"].apply(lambda x: "Duplicate" if x else "New")

                    # ✅ Added metrics
                    st.markdown("### 📈 Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🔢 Total Entries", len(df))
                    col2.metric("🆕 New Materials", len(df[df["Status"] == "New"]))
                    col3.metric("♻️ Duplicates", len(df[df["Status"] == "Duplicate"]))

                    # ✅ Added pie chart
                    import plotly.express as px
                    fig = px.pie(df, names='Status', title='New vs Duplicate Materials')
                    st.plotly_chart(fig)

                    st.success("✅ Deduplication complete!")
                    st.markdown("### 🔎 Deduplicated Results")
                    st.dataframe(df[["System_1", "Predicted_System_2", "Status"]].style.applymap(
                        lambda val: 'background-color: #ffcccc' if val == 'Duplicate' else 'background-color: #ccffcc',
                        subset=['Status']
                    ))

                    df_new_only = df[df["Status"] == "New"]
                    output_file = "new_materials_only.xlsx"
                    df_new_only.to_excel(output_file, index=False)

                    # ✅ Added logging
                    from datetime import datetime
                    log_entry = {
                        "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "New Items": len(df_new_only)
                    }
                    pd.DataFrame([log_entry]).to_csv("history_log.csv", mode='a', index=False)

                    with open(output_file, "rb") as f:
                        st.download_button("📥 Download New Materials Only", f, file_name="new_materials_only.xlsx")
            else:
                st.info("Please upload your ERP file to continue deduplication.")

        # Always offer full download
        full_output = "predicted_output.xlsx"
        df.to_excel(full_output, index=False)
        with open(full_output, "rb") as f:
            st.download_button("📥 Download All Predicted Materials", f, file_name="predicted_output.xlsx")
'''
