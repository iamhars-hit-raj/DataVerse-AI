import json

from backend.ai_analyst import model, dataframe_context


def plan_dashboard(df):

    context = dataframe_context(df)

    prompt = f"""
You are a Senior Business Intelligence Consultant.

Dataset

{context}

Design an executive dashboard.

Return ONLY JSON.

Example:

{{
"title":"Sales Dashboard",

"kpis":[
"Total Sales",
"Average Profit",
"Orders"
],

"charts":[

{{"type":"bar","x":"Category","y":"Sales"}},

{{"type":"line","x":"Date","y":"Sales"}},

{{"type":"pie","x":"Region"}},

{{"type":"heatmap"}}

],

"summary":"Executive summary here.",

"recommendations":[

"Recommendation 1",

"Recommendation 2",

"Recommendation 3"

]
}}

Return ONLY JSON.
"""

    response = model.generate_content(prompt)

    text = response.text

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1:
        text = text[start:end]

    try:
        return json.loads(text)

    except Exception:

        return {
            "title": "Dashboard",
            "kpis": [],
            "charts": [],
            "summary": "",
            "recommendations": []
        }