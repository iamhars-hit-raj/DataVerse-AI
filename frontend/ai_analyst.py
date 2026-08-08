import streamlit as st

from backend.ai_analyst import answer_question
from backend.chart_ai import generate_chart
from backend.ai_insights import generate_ai_insights


def ai_analyst_page(df):

    st.title("🤖 AI Data Analyst")

    st.caption(
        "Chat with your dataset using Gemini AI."
    )

    st.divider()

    # =====================================================
    # AI Executive Insights
    # =====================================================

    st.subheader("📊 AI Executive Insights")

    if st.button(
        "✨ Generate AI Insights",
        use_container_width=True
    ):

        with st.spinner("Gemini is analyzing the entire dataset..."):

            insights = generate_ai_insights(df)

        st.success("Analysis Complete")

        st.markdown(insights)

    st.divider()

    # =====================================================
    # Chat History Initialization
    # =====================================================

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # =====================================================
    # Display Previous Messages
    # =====================================================

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("chart") is not None:
                st.pyplot(message["chart"])

    # =====================================================
    # Chat Input
    # =====================================================

    prompt = st.chat_input(
        "Ask anything about your dataset..."
    )

    if prompt:

        # ---------------- User Message ----------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):

            st.markdown(prompt)

        # ---------------- Assistant ----------------

        with st.chat_message("assistant"):

            with st.spinner("Gemini is analyzing your dataset..."):

                response = answer_question(
                    df,
                    prompt
                )

                st.markdown(response)

                # Generate chart if applicable
                fig = generate_chart(
                    df,
                    prompt
                )

                if fig is not None:

                    st.pyplot(fig)

        # Save assistant response

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response,
                "chart": fig if fig is not None else None
            }
        )

    # =====================================================
    # Clear Chat
    # =====================================================

    st.divider()

    col1, col2 = st.columns([1, 3])

    with col1:

        if st.button(
            "🗑 Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()

    with col2:

        st.info(
            "💡 Try asking:\n\n"
            "- Summarize this dataset\n"
            "- Show correlation heatmap\n"
            "- Show histogram of Age\n"
            "- Recommend preprocessing steps\n"
            "- Which ML model should I use?"
        )