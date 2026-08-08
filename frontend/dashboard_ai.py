import streamlit as st

from backend.dashboard_ai import generate_dashboard


def ai_dashboard_page(df):

    st.title("📊 AI Dashboard")

    st.caption(
        "Automatically generated executive dashboard."
    )

    if st.button(
        "✨ Generate AI Dashboard",
        use_container_width=True
    ):

        with st.spinner("Building dashboard..."):

            dashboard = generate_dashboard(df)

        st.success("Dashboard Ready!")

        st.divider()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", dashboard["rows"])
        c2.metric("Columns", dashboard["columns"])
        c3.metric("Missing", dashboard["missing"])
        c4.metric("Duplicates", dashboard["duplicates"])

        st.divider()

        if dashboard["heatmap"] is not None:

            st.subheader("Correlation Heatmap")

            st.pyplot(dashboard["heatmap"])

        st.divider()

        st.subheader("🤖 Executive Insights")

        st.markdown(
            dashboard["insights"]
        )