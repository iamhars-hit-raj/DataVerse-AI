import streamlit as st

from backend.feature_engineering import split_dataset


def train_test_split_tab(df):

    st.subheader("✂️ Train / Test Split")

    target = st.selectbox(
        "Target Column",
        df.columns,
        key="split_target"
    )

    train_size = st.slider(
        "Training Size",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.05
    )

    random_state = st.number_input(
        "Random State",
        value=42,
        step=1
    )

    shuffle = st.checkbox(
        "Shuffle Dataset",
        value=True
    )

    if st.button(
        "Create Train/Test Split",
        use_container_width=True
    ):

        X_train, X_test, y_train, y_test = split_dataset(
            df,
            target,
            train_size,
            int(random_state),
            shuffle
        )

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.target = target

        st.success("✅ Train/Test Split Created")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Training Rows", len(X_train))

        with col2:
            st.metric("Testing Rows", len(X_test))

        st.write("### X_train Preview")
        st.dataframe(X_train.head(), use_container_width=True)

        st.write("### y_train Preview")
        st.dataframe(y_train.to_frame(), use_container_width=True)