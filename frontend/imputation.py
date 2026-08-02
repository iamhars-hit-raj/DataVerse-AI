import streamlit as st

from backend.feature_engineering import (
    analyze_features,
    impute_numeric,
    impute_categorical,
)


def imputation_tab(df):

    st.subheader("🩹 Missing Value Imputation")

    info = analyze_features(df)

    # ===============================
    # Numeric Columns
    # ===============================

    if info["numeric"]:

        st.markdown("### 🔢 Numeric Columns")

        num_col = st.selectbox(
            "Select Numeric Column",
            info["numeric"],
            key="num_impute_col"
        )

        method = st.radio(
            "Method",
            ["Mean", "Median", "Zero"],
            horizontal=True,
            key="num_method"
        )

        if st.button(
            "Apply Numeric Imputation",
            use_container_width=True
        ):

            updated_df = impute_numeric(
                df,
                num_col,
                method
            )

            st.session_state.df = updated_df

            st.success("✅ Numeric imputation completed!")

    # ===============================
    # Categorical Columns
    # ===============================

    if info["categorical"]:

        st.markdown("### 🅰️ Categorical Columns")

        cat_col = st.selectbox(
            "Select Categorical Column",
            info["categorical"],
            key="cat_impute_col"
        )

        if st.button(
            "Apply Categorical Imputation",
            use_container_width=True
        ):

            updated_df = impute_categorical(
                df,
                cat_col
            )

            st.session_state.df = updated_df

            st.success("✅ Categorical imputation completed!")