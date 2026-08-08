import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")


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
{df.describe(include='all').to_string()}

First Five Rows:
{df.head().to_string()}
"""

    return context


def answer_question(df, question):

    context = dataframe_context(df)

    prompt = f"""
You are an expert Data Scientist.

A user uploaded a dataset.

Dataset Information:

{context}

User Question:

{question}

Instructions:

- Answer only using the dataset information.
- If the answer requires inference, explain it.
- Give professional Data Science recommendations.
- Keep answers concise but informative.
"""

    try:

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:

        return f"❌ Gemini Error:\n\n{e}"