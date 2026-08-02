import streamlit as st

from backend.feature_engineering import (
    analyze_features,
    label_encode,
    one_hot_encode
)


def encoding_tab(df):

    st.subheader("🧬 Encoding")

    info = analyze_features(df)

    if not info["categorical"]:
        st.info("No categorical columns available.")
        return

    column = st.selectbox(
        "Select Categorical Column",
        info["categorical"],
        key="encoding_column"
    )

    method = st.radio(
        "Encoding Method",
        ["Label Encoding", "One-Hot Encoding"],
        horizontal=True,
        key="encoding_method"
    )

    st.write("### Preview")

    st.write(df[column].head())

    if st.button("Apply Encoding", use_container_width=True):

        if method == "Label Encoding":
            updated_df = label_encode(df, column)

        else:
            updated_df = one_hot_encode(df, column)

        st.session_state.df = updated_df

        st.success("✅ Encoding completed!")