from backend.ai_analyst import model

from backend.copilot_prompt import copilot_prompt


def generate_copilot(df):

    prompt = copilot_prompt(df)

    response = model.generate_content(prompt)

    return response.text