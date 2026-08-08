import streamlit as st

from backend.data_health import dataset_health


def show_health_score(df):

    health = dataset_health(df)

    st.header("🩺 Dataset Health")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Health Score",
            f"{health['score']}/100"
        )

        st.metric(
            "Quality",
            health["status"]
        )

    with col2:

        st.metric(
            "Numeric Columns",
            health["numeric"]
        )

        st.metric(
            "Categorical Columns",
            health["categorical"]
        )

    st.progress(
        health["score"] / 100
    )

    st.subheader("Issues Found")

    if health["issues"]:

        for issue in health["issues"]:

            st.write("•", issue)

    else:

        st.success(
            "No major data quality issues found."
        )