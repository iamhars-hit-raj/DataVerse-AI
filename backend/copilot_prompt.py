from backend.ai_analyst import dataframe_context


def copilot_prompt(df):

    context = dataframe_context(df)

    return f"""
You are a Senior Data Scientist.

Dataset:

{context}

Act like an AI Copilot.

Return Markdown only.

Include these sections:

# Dataset Health

Give a score out of 100.

---

# Problems Found

Mention

- Missing values
- Duplicate rows
- Outliers
- Data types
- High cardinality
- Class imbalance
- Multicollinearity

---

# Recommended Data Cleaning

---

# Recommended Feature Engineering

---

# Recommended Machine Learning Model

Explain WHY.

---

# Estimated Accuracy

Give an estimated range.

---

# Business Recommendations

Return only Markdown.
"""