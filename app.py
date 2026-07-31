import streamlit as st
import pandas as pd

from backend.overview import show_dataset_overview
from backend.cleaning import data_cleaning
from backend.eda import exploratory_data_analysis

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="DataVerse AI",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# Title
# ==================================================
st.title("📊 DataVerse AI")
st.subheader("An End-to-End AI-Powered Data Science Platform")

st.write(
    "Welcome! Upload a CSV or Excel dataset to begin."
)

st.divider()

# ==================================================
# File Upload
# ==================================================
uploaded_file = st.file_uploader(
    "📂 Choose a CSV or Excel file",
    type=["csv", "xlsx"]
)

# ==================================================
# Read Dataset
# ==================================================
if uploaded_file is not None:

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # ----------------------------
        # Dataset Overview
        # ----------------------------
        show_dataset_overview(df)

        st.divider()

        # ----------------------------
        # Data Cleaning
        # ----------------------------
        df = data_cleaning(df)
        st.divider()
        exploratory_data_analysis(df)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a CSV or Excel dataset to continue.")