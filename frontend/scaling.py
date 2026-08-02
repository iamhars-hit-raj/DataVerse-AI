import streamlit as st

from backend.feature_engineering import (
    analyze_features,
    scale_features
)


def scaling_tab(df):

    st.subheader("📏 Feature Scaling")

    info = analyze_features(df)

    if not info["numeric"]:
        st.info("No numeric columns available.")
        return

    selected_columns = st.multiselect(
        "Select Numeric Columns",
        info["numeric"]
    )

    method = st.selectbox(
        "Scaling Method",
        [
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler"
        ]
    )

    if st.button(
        "Apply Scaling",
        use_container_width=True
    ):

        updated_df = scale_features(
            df,
            selected_columns,
            method
        )

        st.session_state.df = updated_df

        st.success("✅ Scaling completed successfully!")