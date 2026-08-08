import json

from backend.ai_analyst import model, dataframe_context


def plan_chart(df, question):

    # =====================================================
    # Dataset Context
    # =====================================================

    context = dataframe_context(df)

    # =====================================================
    # Prompt
    # =====================================================

    prompt = f"""
You are an expert Data Visualization assistant.

Dataset:

{context}

User Question:

{question}

Your task is to determine the most appropriate visualization.

Return ONLY valid JSON.

Examples:

{{"chart":"histogram","x":"Age"}}

{{"chart":"scatter","x":"Age","y":"Salary"}}

{{"chart":"line","x":"Date","y":"Sales"}}

{{"chart":"bar","x":"Department","y":"Salary"}}

{{"chart":"box","x":"Salary"}}

{{"chart":"pie","x":"Department"}}

{{"chart":"heatmap"}}

If no visualization is appropriate, return:

{{"chart":"none"}}

Do not return markdown.

Do not use code fences.

Do not explain anything.

Only return JSON.
"""

    # =====================================================
    # Gemini Response
    # =====================================================

    response = model.generate_content(prompt)

    text = response.text.strip()

    # =====================================================
    # Remove Markdown Code Fences
    # =====================================================

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    # =====================================================
    # Extract JSON if Gemini adds extra text
    # =====================================================

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end > start:
        text = text[start:end]

    # =====================================================
    # Parse JSON Safely
    # =====================================================

    try:

        return json.loads(text)

    except Exception:

        return {
            "chart": "none"
        }