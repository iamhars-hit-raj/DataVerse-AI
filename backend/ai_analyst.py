import os

import streamlit as st
from google import genai

from dotenv import load_dotenv


# =====================================================
# Environment Variables
# =====================================================

load_dotenv()


# =====================================================
# Get Gemini API Key
# =====================================================

def get_gemini_api_key():

    # Local development
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    # Streamlit Cloud
    try:

        api_key = st.secrets["GEMINI_API_KEY"]

        if api_key:
            return api_key

    except Exception:
        pass

    return None


# =====================================================
# Gemini API Configuration
# =====================================================

GEMINI_API_KEY = get_gemini_api_key()


if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is not configured. "
        "Add it to your .env file locally or "
        "Streamlit Cloud Secrets when deployed."
    )


# =====================================================
# Gemini Client
# =====================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =====================================================
# Gemini Model Adapter
# =====================================================
#
# This keeps the old interface:
#
#     model.generate_content(prompt)
#
# so the rest of the DataVerse application
# does not need to be rewritten.
#
# =====================================================

class GeminiModel:

    def __init__(self, model_name="gemini-2.5-flash"):

        self.model_name = model_name

    def generate_content(self, prompt):

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response


# =====================================================
# Global Model
# =====================================================

model = GeminiModel(
    "gemini-2.5-flash"
)


# =====================================================
# Dataset Context
# =====================================================

def dataframe_context(df):

    context = f"""
Dataset Shape:

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Columns:

{list(df.columns)}

Data Types:

{df.dtypes.to_string()}

Missing Values:

{df.isnull().sum().to_string()}

Summary Statistics:

{df.describe(include="all").to_string()}

First Five Rows:

{df.head().to_string()}
"""

    return context


# =====================================================
# AI Data Analyst
# =====================================================

def answer_question(df, question):

    context = dataframe_context(df)

    prompt = f"""
You are an expert Data Scientist and AI Data Analyst.

A user uploaded a dataset.

=====================================================
DATASET INFORMATION
=====================================================

{context}

=====================================================
USER QUESTION
=====================================================

{question}

=====================================================
INSTRUCTIONS
=====================================================

- Answer using the dataset information provided above.
- Do not invent facts that are not supported by the dataset.
- If the answer requires inference, clearly explain the inference.
- Provide professional Data Science recommendations when appropriate.
- Use clear Markdown formatting.
- Keep the response concise but informative.
- When numerical values are available, use them.
- When useful, use bullet points or numbered lists.
"""

    try:

        response = model.generate_content(
            prompt
        )

        return response.text

    except Exception as e:

        return (
            "❌ **Gemini Error**\n\n"
            f"`{e}`"
        )