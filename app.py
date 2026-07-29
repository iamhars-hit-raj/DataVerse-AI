import streamlit as st

# Configure the page
st.set_page_config(
    page_title="DataVerse AI",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("📊 DataVerse AI")

# Subtitle
st.subheader("An End-to-End AI-Powered Data Science Platform")

# Welcome message
st.write("""
Welcome to DataVerse AI!

This platform will eventually help users:

- 📁 Upload datasets
- 🧹 Clean data
- 📊 Perform Exploratory Data Analysis (EDA)
- 🤖 Train Machine Learning models
- 📈 Compare model performance
- 🔍 Explain predictions
- 💬 Chat with datasets using AI
""")