import pandas as pd
import streamlit as st

from backend.automl import run_automl
from backend.report import generate_report

from backend.ml import (
    get_model,
    train_model,
    classification_metrics,
    regression_metrics,
)

from backend.model_loader import load_model

from backend.evaluation import (
    plot_confusion_matrix,
    get_classification_report,
    get_feature_importance,
)

from backend.export import (
    save_model,
    prediction_dataframe,
)

from backend.explainability import (
    shap_summary_plot,
)

from frontend.hyperparameters import (
    get_hyperparameters,
)

from frontend.prediction_playground import prediction_playground
from frontend.model_comparison import show_model_comparison
from backend.deployment import export_fastapi_project


# ============================================================
# Target Validation Helpers
# ============================================================

IDENTIFIER_NAMES = {
    "id",
    "patient id",
    "patient_id",
    "patientid",
    "name",
    "doctor",
    "hospital",
    "uuid",
    "user id",
    "user_id",
    "userid",
    "record id",
    "record_id",
}


def get_target_name():
    """
    Try to identify the original target column from session state.

    Different versions of the Feature Engineering page may use
    different session-state keys, so several common names are checked.
    """

    possible_keys = [
        "target_column",
        "target_col",
        "target",
        "target_variable",
        "y_column",
        "label_column",
        "selected_target",
    ]

    for key in possible_keys:
        if key in st.session_state:
            value = st.session_state[key]

            if isinstance(value, str) and value.strip():
                return value

    return None


def validate_target(y, problem_type):
    """
    Validate the target before sending it to scikit-learn.

    Returns:
        (True, information_dict) when the target is safe.
        (False, error_message) when training should be stopped.
    """

    if y is None:
        return False, {
            "error": "Target data is missing."
        }

    # Convert Series/DataFrame safely
    if isinstance(y, pd.DataFrame):

        if y.shape[1] != 1:
            return False, {
                "error": (
                    "The target contains multiple columns. "
                    "Please select exactly one target column."
                )
            }

        y = y.iloc[:, 0]

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    y = y.dropna()

    n_samples = len(y)

    if n_samples == 0:
        return False, {
            "error": "The target contains no usable values."
        }

    n_unique = y.nunique(dropna=True)

    target_name = get_target_name()

    # --------------------------------------------------------
    # Basic class validation
    # --------------------------------------------------------

    if problem_type == "Classification":

        if n_unique < 2:
            return False, {
                "error": (
                    "Classification requires at least 2 classes, "
                    f"but the target contains only {n_unique}."
                )
            }

        # ----------------------------------------------------
        # High-cardinality target detection
        # ----------------------------------------------------

        unique_ratio = n_unique / n_samples

        normalized_name = ""

        if target_name:
            normalized_name = (
                str(target_name)
                .strip()
                .lower()
                .replace("-", " ")
            )

        looks_like_identifier = (
            normalized_name in IDENTIFIER_NAMES
            or normalized_name.endswith(" id")
            or normalized_name.endswith("_id")
            or normalized_name.endswith("id")
        )

        # Strong identifier detection
        if looks_like_identifier and unique_ratio > 0.10:

            return False, {
                "error": (
                    f"'{target_name}' appears to be an identifier or "
                    "high-cardinality field and should not be used as a "
                    "classification target."
                ),
                "target_name": target_name,
                "n_samples": n_samples,
                "n_unique": n_unique,
                "unique_ratio": unique_ratio,
            }

        # Extremely high-cardinality detection.
        #
        # Example:
        # 55,500 rows
        # 49,992 unique Name values
        #
        # This is exactly the situation that caused the previous
        # Logistic Regression problem.
        if unique_ratio >= 0.50:

            return False, {
                "error": (
                    f"The target contains {n_unique:,} unique values "
                    f"among {n_samples:,} samples "
                    f"({unique_ratio:.1%} unique). "
                    "This is too high for a normal classification "
                    "problem and strongly suggests that an identifier "
                    "or free-text column was selected as the target."
                ),
                "target_name": target_name,
                "n_samples": n_samples,
                "n_unique": n_unique,
                "unique_ratio": unique_ratio,
            }

        # ----------------------------------------------------
        # Extremely small class warning
        # ----------------------------------------------------

        class_counts = y.value_counts()

        smallest_class = class_counts.min()

        if smallest_class < 2:

            return False, {
                "error": (
                    "At least one target class contains only one sample. "
                    "Please choose a target with enough observations "
                    "per class."
                ),
                "target_name": target_name,
                "n_samples": n_samples,
                "n_unique": n_unique,
            }

        return True, {
            "target_name": target_name,
            "n_samples": n_samples,
            "n_unique": n_unique,
            "unique_ratio": unique_ratio,
            "class_counts": class_counts,
        }

    # ========================================================
    # Regression Validation
    # ========================================================

    if problem_type == "Regression":

        # Regression targets must be numeric.
        numeric_y = pd.to_numeric(y, errors="coerce")

        invalid_count = numeric_y.isna().sum()

        if invalid_count > 0:

            return False, {
                "error": (
                    "Regression requires a numeric target. "
                    f"{invalid_count:,} target values could not be "
                    "converted to numbers."
                ),
                "target_name": target_name,
            }

        if numeric_y.nunique() < 2:

            return False, {
                "error": (
                    "Regression requires variation in the target. "
                    "The selected target contains only one unique value."
                ),
                "target_name": target_name,
            }

        return True, {
            "target_name": target_name,
            "n_samples": n_samples,
            "n_unique": numeric_y.nunique(),
        }

    return False, {
        "error": "Unknown problem type selected."
    }


def display_target_information(y, problem_type, validation_info):
    """
    Display useful information about the selected target.
    """

    target_name = validation_info.get("target_name")

    st.subheader("🎯 Target Validation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Samples",
        f"{validation_info.get('n_samples', len(y)):,}",
    )

    col2.metric(
        "Unique Values",
        f"{validation_info.get('n_unique', y.nunique()):,}",
    )

    if problem_type == "Classification":

        ratio = validation_info.get("unique_ratio", 0)

        col3.metric(
            "Unique Ratio",
            f"{ratio:.2%}",
        )

        if target_name:
            st.caption(
                f"Target column detected: **{target_name}**"
            )

        class_counts = validation_info.get("class_counts")

        if class_counts is not None:

            with st.expander("View class distribution"):

                class_df = (
                    class_counts
                    .rename("Count")
                    .reset_index()
                )

                class_df.columns = [
                    "Class",
                    "Count",
                ]

                st.dataframe(
                    class_df,
                    width="stretch",
                )

    else:

        if target_name:
            st.caption(
                f"Target column detected: **{target_name}**"
            )


# ============================================================
# Main Machine Learning Page
# ============================================================

def machine_learning_page():

    st.title("🤖 Machine Learning Studio")

    st.caption(
        "Train, evaluate and export Machine Learning models."
    )

    st.divider()

    # ========================================================
    # Check Train/Test Split
    # ========================================================

    required = [
        "X_train",
        "X_test",
        "y_train",
        "y_test",
    ]

    if not all(
        key in st.session_state
        for key in required
    ):

        st.warning(
            "⚠️ Please complete the Train/Test Split "
            "in Feature Engineering first."
        )

        return

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test

    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    # ========================================================
    # Problem Type
    # ========================================================

    st.subheader("🎯 Problem Type")

    problem = st.radio(
        "Select Problem Type",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
    )

    st.divider()

    # ========================================================
    # Validate Target BEFORE Model Training
    # ========================================================

    is_valid, validation_info = validate_target(
        y_train,
        problem,
    )

    if not is_valid:

        st.error(
            "🚫 Model training blocked"
        )

        st.warning(
            validation_info.get(
                "error",
                "The selected target is not suitable for this problem type.",
            )
        )

        target_name = validation_info.get(
            "target_name"
        )

        if target_name:
            st.info(
                f"Current target: **{target_name}**"
            )

        # Give useful advice for high-cardinality targets
        if validation_info.get("n_unique"):

            n_unique = validation_info["n_unique"]
            n_samples = validation_info.get(
                "n_samples",
                len(y_train),
            )

            if n_samples:

                ratio = n_unique / n_samples

                if ratio >= 0.50:

                    st.markdown(
                        """
### ❌ Why training was stopped

The selected target has an extremely large number of
unique values compared with the number of rows.

This usually means a column such as:

- `Name`
- `Doctor`
- `Hospital`
- `Patient ID`
- `User ID`
- another identifier

has accidentally been selected as the prediction target.

These columns should normally be **features or identifiers,
not classification targets**.

### ✅ Recommended targets for this dataset

For classification, try:

- `Test Results`
- `Medical Condition`
- `Admission Type`
- `Gender`
- `Medication`

For regression, try:

- `Billing Amount`
- `Age`

Do not use `Name`, `Doctor`, `Hospital`, or ID columns
as classification targets.
                        """
                    )

        return

    # ========================================================
    # Display Target Information
    # ========================================================

    display_target_information(
        y_train,
        problem,
        validation_info,
    )

    st.divider()

    # ========================================================
    # Algorithm Selection
    # ========================================================

    st.subheader("🤖 Select Algorithm")

    if problem == "Classification":

        models = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "Support Vector Machine",
        ]

    else:

        models = [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
        ]

    model_name = st.selectbox(
        "Algorithm",
        models,
    )

    st.divider()

    # ========================================================
    # Hyperparameters
    # ========================================================

    st.subheader("⚙️ Hyperparameters")

    params = get_hyperparameters(
        model_name
    )

    # ========================================================
    # Logistic Regression Safety Information
    # ========================================================

    if (
        problem == "Classification"
        and model_name == "Logistic Regression"
    ):

        n_classes = y_train.nunique()

        if n_classes >= 3:

            st.info(
                f"ℹ️ Multiclass classification detected "
                f"({n_classes} classes). "
                "Logistic Regression will use a multiclass-compatible "
                "solver."
            )

    st.divider()

    # ========================================================
    # Train Model
    # ========================================================

    if st.button(
        "🚀 Train Model",
        width="stretch",
    ):

        # Revalidate immediately before training.
        #
        # This protects against session-state changes between
        # page renders.
        is_valid, validation_info = validate_target(
            y_train,
            problem,
        )

        if not is_valid:

            st.error(
                "🚫 Training cancelled."
            )

            st.warning(
                validation_info.get(
                    "error",
                    "Invalid target.",
                )
            )

            return

        # ----------------------------------------------------
        # Safety information
        # ----------------------------------------------------

        if problem == "Classification":

            n_classes = y_train.nunique()

            if n_classes >= 20:

                st.warning(
                    f"⚠️ This is a multiclass problem with "
                    f"{n_classes} classes. Training may take longer "
                    "than a binary classification problem."
                )

        with st.spinner(
            "Training model..."
        ):

            try:

                model = get_model(
                    problem,
                    model_name,
                    params,
                )

                model = train_model(
                    model,
                    X_train,
                    y_train,
                )

                st.session_state.model = model

                # Store information about the model
                st.session_state.model_problem = problem
                st.session_state.model_name = model_name

                st.success(
                    "✅ Model Trained Successfully!"
                )

            except ValueError as e:

                st.error(
                    "❌ Model training failed."
                )

                st.exception(e)

                st.info(
                    "Check that the selected problem type matches "
                    "the target data and that the target is not an "
                    "identifier column."
                )

                return

            except MemoryError:

                st.error(
                    "❌ The model ran out of memory during training."
                )

                st.warning(
                    "This usually happens when the dataset contains "
                    "too many features/classes or an inappropriate "
                    "high-cardinality target."
                )

                return

            except Exception as e:

                st.error(
                    "❌ Unexpected error while training the model."
                )

                st.exception(e)

                return

    # ========================================================
    # No Model Yet
    # ========================================================

    if "model" not in st.session_state:

        st.info(
            "👆 Select an algorithm and click **Train Model** "
            "to continue."
        )

        return

    model = st.session_state.model

    # ========================================================
    # Metrics
    # ========================================================

    metrics = {}

    st.divider()

    st.subheader(
        "📊 Performance Metrics"
    )

    try:

        if problem == "Classification":

            metrics = classification_metrics(
                model,
                X_test,
                y_test,
            )

        else:

            metrics = regression_metrics(
                model,
                X_test,
                y_test,
            )

        st.session_state.metrics = metrics

    except Exception as e:

        st.error(
            "❌ Unable to calculate performance metrics."
        )

        st.exception(e)

        return

    if metrics:

        cols = st.columns(
            len(metrics)
        )

        for col, (metric, value) in zip(
            cols,
            metrics.items(),
        ):

            try:

                col.metric(
                    metric,
                    f"{value:.4f}",
                )

            except (TypeError, ValueError):

                col.metric(
                    metric,
                    str(value),
                )

    # ========================================================
    # Prediction Preview
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 Prediction Preview"
    )

    try:

        predictions = model.predict(
            X_test
        )

        preview = prediction_dataframe(
            X_test,
            y_test,
            predictions,
        )

        # ----------------------------------------------------
        # Arrow-safe display dataframe
        # ----------------------------------------------------
        #
        # Streamlit/PyArrow can fail when an object column
        # contains mixed types, for example:
        #
        # Name:
        #     123
        #     "DAvId muNoZ"
        #
        # Convert object columns to strings only for display.
        #
        display_preview = preview.copy()

        for col in display_preview.columns:

            if display_preview[col].dtype == "object":

                display_preview[col] = (
                    display_preview[col]
                    .astype("string")
                )

        st.dataframe(
            display_preview.head(20),
            width="stretch",
        )

        st.success(
            f"Showing first "
            f"{min(20, len(preview)):,} predictions."
        )

    except Exception as e:

        st.error(
            "❌ Unable to generate prediction preview."
        )

        st.exception(e)

        preview = pd.DataFrame()

    # ========================================================
    # Classification Evaluation
    # ========================================================

    if (
        problem == "Classification"
        and not preview.empty
    ):

        st.divider()

        st.subheader(
            "📊 Confusion Matrix"
        )

        try:

            fig = plot_confusion_matrix(
                model,
                X_test,
                y_test,
            )

            st.pyplot(fig)

        except Exception as e:

            st.warning(
                "Unable to generate confusion matrix."
            )

            st.exception(e)

        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📋 Classification Report"
        )

        try:

            report = get_classification_report(
                model,
                X_test,
                y_test,
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            # Arrow-safe report
            for col in report_df.columns:

                if report_df[col].dtype == "object":

                    report_df[col] = (
                        report_df[col]
                        .astype("string")
                    )

            st.dataframe(
                report_df,
                width="stretch",
            )

        except Exception as e:

            st.warning(
                "Unable to generate classification report."
            )

            st.exception(e)

    # ========================================================
    # Feature Importance
    # ========================================================

    try:

        importance = get_feature_importance(
            model,
            X_train.columns,
        )

    except Exception:

        importance = None

    if importance is not None:

        st.divider()

        st.subheader(
            "🌳 Feature Importance"
        )

        # Arrow-safe feature importance dataframe
        display_importance = importance.copy()

        for col in display_importance.columns:

            if display_importance[col].dtype == "object":

                display_importance[col] = (
                    display_importance[col]
                    .astype("string")
                )

        st.dataframe(
            display_importance,
            width="stretch",
        )

        try:

            st.bar_chart(
                importance.set_index(
                    "Feature"
                )
            )

        except Exception:

            pass

    # ========================================================
    # Download Predictions
    # ========================================================

    if not preview.empty:

        st.divider()

        st.subheader(
            "📥 Download Predictions"
        )

        csv = preview.to_csv(
            index=False
        )

        st.download_button(
            label="⬇ Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            width="stretch",
        )

    # ========================================================
    # Download Trained Model
    # ========================================================

    st.divider()

    st.subheader(
        "💾 Download Trained Model"
    )

    try:

        model_path = save_model(
            model
        )

        with open(
            model_path,
            "rb",
        ) as file:

            st.download_button(
                label="⬇ Download Model (.pkl)",
                data=file.read(),
                file_name="trained_model.pkl",
                mime="application/octet-stream",
                width="stretch",
            )

    except Exception as e:

        st.warning(
            "Unable to prepare the model download."
        )

        st.exception(e)

    # ========================================================
    # AutoML Leaderboard
    # ========================================================

    st.divider()

    st.header(
        "🏆 AutoML Leaderboard"
    )

    st.caption(
        "Automatically train and compare multiple "
        "Machine Learning models."
    )

    if st.button(
        "🚀 Train All Models",
        width="stretch",
    ):

        with st.spinner(
            "Training multiple models..."
        ):

            try:

                leaderboard = run_automl(
                    problem,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )

                st.success(
                    "✅ AutoML Completed!"
                )

                show_model_comparison(
                    leaderboard
                )

            except Exception as e:

                st.error(
                    "❌ AutoML training failed."
                )

                st.exception(e)

    # ========================================================
    # Explainable AI (SHAP)
    # ========================================================

    st.divider()

    st.header(
        "🧠 Explainable AI"
    )

    st.caption(
        "Understand why your Machine Learning "
        "model made its predictions."
    )

    try:

        fig = shap_summary_plot(
            model,
            X_train,
        )

        if fig is not None:

            st.pyplot(fig)

        else:

            st.info(
                "SHAP visualization is currently supported "
                "for tree-based models such as Decision Trees "
                "and Random Forests."
            )

    except Exception as e:

        st.info(
            "SHAP visualization is not available "
            "for this model."
        )

    # ========================================================
    # PDF Report
    # ========================================================

    st.divider()

    st.header(
        "📄 PDF Report"
    )

    st.caption(
        "Generate a professional PDF report containing "
        "model performance and dataset summary."
    )

    if st.button(
        "📄 Generate PDF Report",
        width="stretch",
    ):

        try:

            report_path = generate_report(
                st.session_state.df,
                st.session_state.metrics,
            )

            st.success(
                "✅ Report Generated Successfully!"
            )

            with open(
                report_path,
                "rb",
            ) as file:

                st.download_button(
                    label="⬇ Download PDF Report",
                    data=file.read(),
                    file_name="DataVerse_Report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

        except Exception as e:

            st.error(
                "❌ Unable to generate PDF report."
            )

            st.exception(e)

    # ========================================================
    # Load Existing Model
    # ========================================================

    st.divider()

    st.header(
        "📂 Load Existing Model"
    )

    uploaded_model = st.file_uploader(
        "Upload a trained .pkl model",
        type=["pkl"],
    )

    if uploaded_model is not None:

        import tempfile

        try:

            with tempfile.NamedTemporaryFile(
                delete=False
            ) as tmp:

                tmp.write(
                    uploaded_model.read()
                )

                loaded_model = load_model(
                    tmp.name
                )

            st.session_state.model = (
                loaded_model
            )

            model = loaded_model

            st.success(
                "✅ Model Loaded Successfully"
            )

        except Exception as e:

            st.error(
                "❌ Unable to load the model."
            )

            st.exception(e)

    # ========================================================
    # Prediction Playground
    # ========================================================

    st.divider()

    st.header(
        "🎮 Prediction Playground"
    )

    try:

        prediction_playground(
            model,
            X_train,
        )

    except Exception as e:

        st.warning(
            "Prediction Playground could not be loaded."
        )

        st.exception(e)

    # ========================================================
    # Model Deployment
    # ========================================================

    st.divider()

    st.header(
        "🚀 Model Deployment"
    )

    st.caption(
        "Export your trained model as a ready-to-run "
        "FastAPI project."
    )

    if st.button(
        "📦 Export Deployment Package",
        width="stretch",
    ):

        with st.spinner(
            "Generating deployment package..."
        ):

            try:

                zip_path = export_fastapi_project(
                    model,
                    X_train.columns.tolist(),
                )

                st.success(
                    "Deployment package generated successfully!"
                )

                with open(
                    zip_path,
                    "rb",
                ) as file:

                    st.download_button(
                        label="⬇ Download Deployment ZIP",
                        data=file.read(),
                        file_name="DataVerse_Deployment.zip",
                        mime="application/zip",
                        width="stretch",
                    )

            except Exception as e:

                st.error(
                    "❌ Unable to generate deployment package."
                )

                st.exception(e)