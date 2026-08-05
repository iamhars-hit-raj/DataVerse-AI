import streamlit as st

from backend.ml import (
    get_model,
    train_model,
    classification_metrics,
    regression_metrics
)

from frontend.hyperparameters import get_hyperparameters


def machine_learning_page():

    st.title("🤖 Machine Learning Studio")

    st.caption(
        "Train and evaluate Machine Learning models."
    )

    st.divider()

    # =====================================================
    # Check Train/Test Split
    # =====================================================

    required = [
        "X_train",
        "X_test",
        "y_train",
        "y_test"
    ]

    if not all(k in st.session_state for k in required):

        st.warning(
            "⚠️ Please complete the Train/Test Split in the Feature Engineering page first."
        )

        return

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    # =====================================================
    # Problem Type
    # =====================================================

    st.subheader("🎯 Problem Type")

    problem = st.radio(
        "Select Problem Type",
        [
            "Classification",
            "Regression"
        ],
        horizontal=True
    )

    st.divider()

    # =====================================================
    # Algorithm Selection
    # =====================================================

    st.subheader("🤖 Select Algorithm")

    if problem == "Classification":

        models = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "Support Vector Machine"
        ]

    else:

        models = [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ]

    model_name = st.selectbox(
        "Algorithm",
        models
    )

    st.divider()

    # =====================================================
    # Hyperparameters
    # =====================================================

    st.subheader("⚙️ Hyperparameters")

    params = get_hyperparameters(model_name)

    st.divider()

    # =====================================================
    # Train Model
    # =====================================================

    if st.button(
        "🚀 Train Model",
        use_container_width=True
    ):

        with st.spinner("Training model..."):

            model = get_model(
                problem,
                model_name,
                params
            )

            model = train_model(
                model,
                X_train,
                y_train
            )

            st.session_state.model = model

        st.success("✅ Model Trained Successfully!")

        st.divider()

        # =====================================================
        # Metrics
        # =====================================================

        st.subheader("📊 Performance Metrics")

        if problem == "Classification":

            metrics = classification_metrics(
                model,
                X_test,
                y_test
            )

        else:

            metrics = regression_metrics(
                model,
                X_test,
                y_test
            )

        cols = st.columns(len(metrics))

        for col, (metric, value) in zip(cols, metrics.items()):

            col.metric(
                metric,
                f"{value:.4f}"
            )

        st.divider()

        # =====================================================
        # Prediction Preview
        # =====================================================

        st.subheader("🔍 Prediction Preview")

        predictions = model.predict(X_test)

        preview = X_test.copy()

        preview["Actual"] = y_test.values
        preview["Predicted"] = predictions

        st.dataframe(
            preview.head(20),
            use_container_width=True
        )

        st.success(
            f"Showing first {min(20, len(preview))} predictions."
        )