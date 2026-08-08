from backend.ai_analyst import model, dataframe_context


def generate_ai_insights(df):

    context = dataframe_context(df)

    prompt = f"""
You are a Senior Data Scientist.

Dataset Information:

{context}

Generate a professional report using these sections:

# Executive Summary

# Data Quality Analysis

# Statistical Insights

# Machine Learning Recommendations

# Business Insights

# Suggested Next Steps

Use markdown formatting.

Be concise.

Do not invent facts.

Base every statement on the dataset.
"""

    response = model.generate_content(prompt)

    return response.text