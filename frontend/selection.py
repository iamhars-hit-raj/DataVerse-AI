import streamlit as st

from backend.feature_engineering import select_features


def feature_selection_tab(df):

    st.subheader("🎯 Feature Selection")

    target = st.selectbox(
        "Target Column",
        df.columns,
        key="fs_target"
    )

    feature_options = [c for c in df.columns if c != target]

    selected_features = st.multiselect(
        "Input Features",
        feature_options,
        default=feature_options
    )

    st.write("### Selected Dataset Preview")

    preview = df[selected_features + [target]]

    st.dataframe(preview.head(), use_container_width=True)

    if st.button(
        "Apply Feature Selection",
        use_container_width=True
    ):

        updated_df = select_features(
            df,
            target,
            selected_features
        )

        st.session_state.df = updated_df
        st.session_state.target = target
        st.session_state.features = selected_features

        st.success("✅ Feature selection applied successfully!")