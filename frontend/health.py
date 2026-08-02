import streamlit as st


def show_health(metrics):

    st.subheader("📊 Dataset Health")

    c1, c2 = st.columns(2)

    with c1:

        st.metric("Health Score", f"{metrics['score']}/100")

        st.metric("Quality", metrics["status"])

    with c2:

        st.metric("Numeric Columns", metrics["numeric"])

        st.metric("Categorical Columns", metrics["categorical"])

    st.progress(metrics["score"] / 100)