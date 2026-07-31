import streamlit as st
import pandas as pd
import plotly.express as px


def exploratory_data_analysis(df):

    st.header("📊 Exploratory Data Analysis")

    analysis = st.selectbox(
        "Select Analysis",
        [
            "Dataset Summary",
            "Descriptive Statistics",
            "Histogram",
            "Box Plot",
            "Correlation Heatmap"
        ]
    )

    # =====================================================
    # Dataset Summary
    # =====================================================
    if analysis == "Dataset Summary":

        summary = pd.DataFrame({
            "Property": [
                "Rows",
                "Columns",
                "Duplicate Rows",
                "Missing Values"
            ],
            "Value": [
                df.shape[0],
                df.shape[1],
                df.duplicated().sum(),
                df.isnull().sum().sum()
            ]
        })

        st.dataframe(summary, use_container_width=True, hide_index=True)

    # =====================================================
    # Descriptive Statistics
    # =====================================================
    elif analysis == "Descriptive Statistics":

        st.dataframe(
            df.describe(include="all"),
            use_container_width=True
        )

    # =====================================================
    # Histogram
    # =====================================================
    elif analysis == "Histogram":

        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if not numeric_columns:
            st.warning("No numeric columns found.")
            return

        column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        bins = st.slider(
            "Number of Bins",
            5,
            100,
            30
        )

        fig = px.histogram(
            df,
            x=column,
            nbins=bins,
            title=f"Histogram of {column}"
        )

        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Box Plot
    # =====================================================
    elif analysis == "Box Plot":

        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if not numeric_columns:
            st.warning("No numeric columns found.")
            return

        column = st.selectbox(
            "Select Numeric Column",
            numeric_columns,
            key="box"
        )

        fig = px.box(
            df,
            y=column,
            points="outliers",
            title=f"Box Plot of {column}"
        )

        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Correlation Heatmap
    # =====================================================
    elif analysis == "Correlation Heatmap":

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            st.warning("Need at least two numeric columns.")
            return

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Correlation Heatmap"
        )

        fig.update_layout(
            template="plotly_white",
            height=700
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info("""
### Interpretation

**+1.00** → Strong Positive Correlation

**0.00** → No Correlation

**-1.00** → Strong Negative Correlation
""")