import streamlit as st

from backend.report_summary import build_report_summary
from backend.report import generate_report


def reports_page(df):

    st.title("📄 Reports Center")

    st.caption(
        "Generate professional reports for your dataset and Machine Learning results."
    )

    st.divider()

    # =====================================================
    # Get Metrics (if available)
    # =====================================================

    metrics = st.session_state.get("metrics", None)

    summary = build_report_summary(
        df,
        metrics
    )

    # =====================================================
    # Dataset Summary
    # =====================================================

    st.subheader("📊 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        summary["rows"]
    )

    col2.metric(
        "Columns",
        summary["columns"]
    )

    col3.metric(
        "Missing Values",
        summary["missing"]
    )

    col4.metric(
        "Duplicates",
        summary["duplicates"]
    )

    st.divider()

    # =====================================================
    # ML Metrics
    # =====================================================

    st.subheader("🤖 Model Performance")

    if summary["metrics"]:

        cols = st.columns(len(summary["metrics"]))

        for col, (metric, value) in zip(
            cols,
            summary["metrics"].items()
        ):

            try:

                col.metric(
                    metric,
                    f"{value:.4f}"
                )

            except Exception:

                col.metric(
                    metric,
                    value
                )

    else:

        st.info(
            "Train a Machine Learning model to display performance metrics."
        )

    st.divider()

    # =====================================================
    # AI Insights
    # =====================================================

    st.subheader("🧠 AI Executive Insights")

    st.markdown(
        summary["insights"]
    )

    st.divider()

    # =====================================================
    # Generate PDF
    # =====================================================

    st.subheader("📥 Download Report")

    if st.button(
        "📄 Generate PDF Report",
        use_container_width=True
    ):

        with st.spinner(
            "Generating report..."
        ):

            report_path = generate_report(
                df,
                summary["metrics"]
            )

        with open(
            report_path,
            "rb"
        ) as file:

            st.download_button(
                label="⬇ Download PDF",
                data=file.read(),
                file_name="DataVerse_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.divider()

    st.caption(
        f"🕒 Generated on: {summary['generated_on']}"
    )