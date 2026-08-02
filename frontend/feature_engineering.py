import streamlit as st

from backend.feature_engineering import analyze_features

from frontend.imputation import imputation_tab
from frontend.encoding import encoding_tab
from frontend.scaling import scaling_tab
from frontend.selection import feature_selection_tab
from frontend.split import train_test_split_tab

def feature_engineering_page(df):

    st.title("🧪 Feature Engineering Studio")

    st.caption("Prepare your dataset before Machine Learning.")

    st.divider()

    # =====================================================
    # Analyze Dataset
    # =====================================================

    info = analyze_features(df)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔢 Numeric Columns")

        if info["numeric"]:
            st.write(info["numeric"])
        else:
            st.info("No numeric columns found.")

        st.subheader("🅰️ Categorical Columns")

        if info["categorical"]:
            st.write(info["categorical"])
        else:
            st.info("No categorical columns found.")

    with col2:

        st.subheader("📅 Datetime Columns")

        if info["datetime"]:
            st.write(info["datetime"])
        else:
            st.info("No datetime columns found.")

        st.subheader("✅ Boolean Columns")

        if info["boolean"]:
            st.write(info["boolean"])
        else:
            st.info("No boolean columns found.")

    st.divider()

    # =====================================================
    # Target Column Selection
    # =====================================================

    st.subheader("🎯 Select Target Column")

    target = st.selectbox(
        "Choose the target variable",
        df.columns,
        key="target_column"
    )

    st.success(f"Selected Target: **{target}**")

    st.divider()

    # =====================================================
    # Dataset Summary
    # =====================================================

    st.subheader("📋 Dataset Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.metric("Rows", f"{df.shape[0]:,}")
        st.metric("Columns", df.shape[1])

    with summary_col2:
        st.metric("Missing Values", int(df.isnull().sum().sum()))
        st.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.divider()

    # =====================================================
    # Feature Engineering Tools
    # =====================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🩹 Missing Values",
            "🧬 Encoding",
            "📏 Scaling",
            "🎯 Feature Selection",
            "✂️ Train/Test Split"
        ]
    )

    with tab1:
        imputation_tab(df)

    with tab2:
        encoding_tab(df)

    with tab3:
        scaling_tab(df)

    with tab4:
        feature_selection_tab(df)
        
    with tab5:
        train_test_split_tab(df)