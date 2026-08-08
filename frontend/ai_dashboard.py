import streamlit as st

from backend.dashboard_ai import generate_dashboard


def ai_dashboard_page(df):

    st.title("📈 AI Dashboard")

    st.caption(
        "Generate an executive dashboard powered by Gemini AI."
    )

    st.divider()

    if st.button(
        "✨ Generate AI Dashboard",
        use_container_width=True
    ):

        with st.spinner(
            "Gemini is building your executive dashboard..."
        ):

            dashboard = generate_dashboard(df)

        st.success("Dashboard Generated!")

        # =====================================
        # KPI Cards
        # =====================================

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rows",
            dashboard["rows"]
        )

        col2.metric(
            "Columns",
            dashboard["columns"]
        )

        col3.metric(
            "Missing",
            dashboard["missing"]
        )

        col4.metric(
            "Duplicates",
            dashboard["duplicates"]
        )

        st.divider()

        # =====================================
        # AI Title
        # =====================================

        if dashboard.get("title"):
            st.header(dashboard["title"])

        # =====================================
        # Executive Summary
        # =====================================

        if dashboard.get("summary"):
            st.subheader("📝 Executive Summary")
            st.write(dashboard["summary"])

        # =====================================
        # AI Charts
        # =====================================

        charts = dashboard.get("charts", [])

        if charts:

            st.subheader("📊 AI Visualizations")

            for fig in charts:

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # =====================================
        # Correlation Heatmap
        # =====================================

        if dashboard["heatmap"] is not None:

            st.subheader("🔥 Correlation Heatmap")

            st.pyplot(
                dashboard["heatmap"]
            )

        # =====================================
        # AI Recommendations
        # =====================================

        recommendations = dashboard.get(
            "recommendations",
            []
        )

        if recommendations:

            st.subheader("💡 Recommendations")

            for item in recommendations:

                st.success(item)

        # =====================================
        # AI Insights
        # =====================================

        st.subheader("🧠 Executive Insights")

        st.markdown(
            dashboard["insights"]
        )