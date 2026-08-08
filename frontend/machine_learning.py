from xml.parsers.expat import model

import pandas as pd
import streamlit as st

from backend.automl import run_automl
from backend.report import generate_report

from backend.ml import (
    get_model,
    train_model,
    classification_metrics,
    regression_metrics
)
from backend.model_loader import load_model

from backend.evaluation import (
    plot_confusion_matrix,
    get_classification_report,
    get_feature_importance
)

from backend.export import (
    save_model,
    prediction_dataframe
)

from backend.explainability import (
    shap_summary_plot
)

from frontend.hyperparameters import (
    get_hyperparameters
)
from frontend.prediction_playground import prediction_playground
from frontend.model_comparison import show_model_comparison
from backend.deployment import export_fastapi_project

def machine_learning_page():

    st.title("🤖 Machine Learning Studio")

    st.caption(
        "Train, evaluate and export Machine Learning models."
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
            "⚠️ Please complete the Train/Test Split in Feature Engineering first."
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

    params = get_hyperparameters(
        model_name
    )

    st.divider()

    # =====================================================
    # Train Model
    # =====================================================

    if st.button(
        "🚀 Train Model",
        width="stretch"
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

        st.success(
            "✅ Model Trained Successfully!"
        )

    if "model" not in st.session_state:
        return

    model = st.session_state.model

    metrics = {}

    # =====================================================
    # Metrics
    # =====================================================

    st.divider()

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

    st.session_state.metrics = metrics

    cols = st.columns(len(metrics))

    for col, (metric, value) in zip(
        cols,
        metrics.items()
    ):

        col.metric(
            metric,
            f"{value:.4f}"
        )

    # =====================================================
    # Prediction Preview
    # =====================================================

    st.divider()

    st.subheader("🔍 Prediction Preview")

    predictions = model.predict(
        X_test
    )

    preview = prediction_dataframe(
        X_test,
        y_test,
        predictions
    )

    st.dataframe(
        preview.head(20),
        width="stretch"
    )

    st.success(
        f"Showing first {min(20, len(preview))} predictions."
    )

    # =====================================================
    # Classification Evaluation
    # =====================================================

    if problem == "Classification":

        st.divider()

        st.subheader(
            "📊 Confusion Matrix"
        )

        fig = plot_confusion_matrix(
            model,
            X_test,
            y_test
        )

        st.pyplot(fig)

        st.divider()

        st.subheader(
            "📋 Classification Report"
        )

        report = get_classification_report(
            model,
            X_test,
            y_test
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            width="stretch"
        )

    # =====================================================
    # Feature Importance
    # =====================================================

    importance = get_feature_importance(
        model,
        X_train.columns
    )

    if importance is not None:

        st.divider()

        st.subheader(
            "🌳 Feature Importance"
        )

        st.dataframe(
            importance,
            width="stretch"
        )

        st.bar_chart(
            importance.set_index(
                "Feature"
            )
        )
    # =====================================================
    # Download Predictions
    # =====================================================

    st.divider()

    st.subheader("📥 Download Predictions")

    csv = preview.to_csv(index=False)

    st.download_button(
        label="⬇ Download Predictions CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
        width="stretch"
    )

    # =====================================================
    # Download Trained Model
    # =====================================================

    st.divider()

    st.subheader("💾 Download Trained Model")

    model_path = save_model(model)

    with open(model_path, "rb") as file:

        st.download_button(
            label="⬇ Download Model (.pkl)",
            data=file.read(),
            file_name="trained_model.pkl",
            mime="application/octet-stream",
            width="stretch"
        )

    # =====================================================
    # AutoML Leaderboard
    # =====================================================

    st.divider()

    st.header("🏆 AutoML Leaderboard")

    st.caption(
        "Automatically train and compare multiple Machine Learning models."
    )

    if st.button(
        "🚀 Train All Models",
        width="stretch"
    ):

        with st.spinner("Training multiple models..."):

            leaderboard = run_automl(
                problem,
                X_train,
                X_test,
                y_train,
                y_test
            )

        st.success("✅ AutoML Completed!")

        show_model_comparison(leaderboard)

    # =====================================================
    # Explainable AI (SHAP)
    # =====================================================

    st.divider()

    st.header("🧠 Explainable AI")

    st.caption(
        "Understand why your Machine Learning model made its predictions."
    )

    fig = shap_summary_plot(
        model,
        X_train
    )

    if fig is not None:

        st.pyplot(fig)

    else:

        st.info(
            "SHAP visualization is currently supported for tree-based models such as Decision Trees and Random Forests."
        )

    # =====================================================
    # PDF Report
    # =====================================================

    st.divider()

    st.header("📄 PDF Report")

    st.caption(
        "Generate a professional PDF report containing model performance and dataset summary."
    )

    if st.button(
        "📄 Generate PDF Report",
        width="stretch"
    ):

        report_path = generate_report(
            st.session_state.df,
            st.session_state.metrics
        )

        st.success(
            "✅ Report Generated Successfully!"
        )

        with open(report_path, "rb") as file:

            st.download_button(
                label="⬇ Download PDF Report",
                data=file.read(),
                file_name="DataVerse_Report.pdf",
                mime="application/pdf",
                width="stretch"
            )
    st.divider()

    st.header("📂 Load Existing Model")

    uploaded_model = st.file_uploader(
        "Upload a trained .pkl model",
        type=["pkl"]
    )

    if uploaded_model is not None:

        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded_model.read())

            model = load_model(tmp.name)

            st.session_state.model = model

        st.success("✅ Model Loaded Successfully")
        
    # =====================================================
    # Prediction Playground
    # =====================================================

    prediction_playground(
        model,
        X_train
    )
    # =====================================================
    # Model Deployment
    # =====================================================

    st.divider()

    st.header("🚀 Model Deployment")

    st.caption(
        "Export your trained model as a ready-to-run FastAPI project."
    )

    if st.button(
        "📦 Export Deployment Package",
        use_container_width=True
    ):

        with st.spinner(
            "Generating deployment package..."
        ):

            zip_path = export_fastapi_project(
                model,
                X_train.columns.tolist()
            )

        st.success(
            "Deployment package generated successfully!"
        )

        with open(zip_path, "rb") as file:

            st.download_button(
                label="⬇ Download Deployment ZIP",
                data=file.read(),
                file_name="DataVerse_Deployment.zip",
                mime="application/zip",
                use_container_width=True
            )