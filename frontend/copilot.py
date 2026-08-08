import streamlit as st

from backend.copilot import generate_copilot


def copilot_page(df):

    st.title("🧠 AI Copilot")

    st.caption(
        "Your personal AI Data Scientist."
    )

    st.divider()

    st.info(
        "Ask the AI to evaluate your dataset and recommend the best next steps."
    )

    if st.button(
        "🚀 Analyze Dataset",
        use_container_width=True
    ):

        with st.spinner(
            "AI Copilot is analyzing your dataset..."
        ):

            report = generate_copilot(df)

        st.success("Analysis Complete!")

        st.markdown(report)

    st.divider()

    st.subheader("💬 Example Questions")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
- Is this dataset ready for ML?

- Which columns should I remove?

- Which model should I use?

- Is my target imbalanced?
            """
        )

    with col2:

        st.info(
            """
- Which preprocessing is required?

- Is feature engineering needed?

- Are there outliers?

- How can accuracy improve?
            """
        )