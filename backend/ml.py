# =====================================================
# Machine Learning Backend
# =====================================================

import inspect

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# Model Factory
# =====================================================

def get_model(problem_type, model_name, params=None):

    params = params or {}

    if problem_type == "Classification":

        if model_name == "Logistic Regression":

            allowed = {
                "C",
                "max_iter",
                "tol",
                "class_weight"
            }

            filtered_params = {
                key: value
                for key, value in params.items()
                if key in allowed
            }

            # Always use saga because it supports
            # multiclass classification and sparse
            # one-hot encoded features.
            filtered_params["solver"] = "saga"

            if "max_iter" not in filtered_params:
                filtered_params["max_iter"] = 1000

            return LogisticRegression(
                **filtered_params
            )

        elif model_name == "Decision Tree":

            allowed = {
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "criterion"
            }

            filtered_params = {
                key: value
                for key, value in params.items()
                if key in allowed
            }

            return DecisionTreeClassifier(
                **filtered_params
            )

        elif model_name == "Random Forest":

            allowed = {
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "criterion"
            }

            filtered_params = {
                key: value
                for key, value in params.items()
                if key in allowed
            }

            return RandomForestClassifier(
                **filtered_params
            )

        elif model_name == "KNN":

            allowed = {
                "n_neighbors",
                "weights",
                "algorithm"
            }

            filtered_params = {
                key: value
                for key, value in params.items()
                if key in allowed
            }

            return KNeighborsClassifier(
                **filtered_params
            )

        elif model_name == "Support Vector Machine":

            allowed = {
                "C",
                "kernel",
                "gamma"
            }

            filtered_params = {
                key: value
                for key, value in params.items()
                if key in allowed
            }

            return SVC(
                **filtered_params
            )

    else:

        if model_name == "Linear Regression":

            return LinearRegression()

        elif model_name == "Decision Tree":

            allowed = {
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "criterion"
            }

            filtered_params = {
                key: value
                for key, value in params.items()
                if key in allowed
            }

            return DecisionTreeRegressor(
                **filtered_params
            )

        elif model_name == "Random Forest":

            allowed = {
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "criterion"
            }

            filtered_params = {
                key: value
                for key, value in params.items()
                if key in allowed
            }

            return RandomForestRegressor(
                **filtered_params
            )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )

# =====================================================
# Build Preprocessing Pipeline
# =====================================================

def _build_preprocessor(X):

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # =================================================
    # Detect Numeric Columns
    # =================================================

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # =================================================
    # Detect Categorical Columns
    # =================================================

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # =================================================
    # Keep Only Reasonable-Cardinality Categories
    # =================================================

    MAX_CATEGORIES = 50

    low_cardinality_columns = []
    high_cardinality_columns = []

    for col in categorical_columns:

        unique_count = X[col].nunique(
            dropna=True
        )

        if unique_count <= MAX_CATEGORIES:

            low_cardinality_columns.append(col)

        else:

            high_cardinality_columns.append(col)

    # =================================================
    # Numeric Pipeline
    # =================================================

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    # =================================================
    # Low-Cardinality Categorical Pipeline
    # =================================================

    if low_cardinality_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                low_cardinality_columns
            )
        )

    # =================================================
    # High-Cardinality Columns
    # =================================================
    #
    # These are intentionally excluded.
    #
    # Example:
    # Name
    # Doctor
    # Hospital
    #
    # Encoding thousands of unique values can create
    # millions of unnecessary features.
    # =================================================

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
    """
    Automatically detects numerical and categorical
    columns and creates the appropriate preprocessing.
    """

    # -------------------------------------------------
    # Make sure input is a DataFrame
    # -------------------------------------------------

    if not isinstance(X, pd.DataFrame):

        X = pd.DataFrame(X)

    # -------------------------------------------------
    # Detect columns
    # -------------------------------------------------

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    transformers = []

    # -------------------------------------------------
    # Numeric preprocessing
    # -------------------------------------------------

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    # -------------------------------------------------
    # Categorical preprocessing
    # -------------------------------------------------

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    # -------------------------------------------------
    # Create ColumnTransformer
    # -------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


# =====================================================
# Train Model
# =====================================================

def train_model(
    model,
    X_train,
    y_train
):
    """
    Automatically preprocess categorical/numeric
    features and train the selected model.
    """

    # =================================================
    # Validate Training Data
    # =================================================

    if X_train is None:

        raise ValueError(
            "Training features are missing."
        )

    if y_train is None:

        raise ValueError(
            "Training target is missing."
        )

    # =================================================
    # Ensure DataFrame
    # =================================================

    if not isinstance(X_train, pd.DataFrame):

        X_train = pd.DataFrame(X_train)

    # =================================================
    # Build Preprocessor
    # =================================================

    preprocessor = _build_preprocessor(
        X_train
    )

    # =================================================
    # Create Complete Pipeline
    # =================================================

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # =================================================
    # Train
    # =================================================

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline


# =====================================================
# Classification Metrics
# =====================================================

def classification_metrics(
    model,
    X_test,
    y_test
):
    """
    Calculate classification performance metrics.
    """

    predictions = model.predict(
        X_test
    )

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "F1 Score":
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
    }

    return metrics


# =====================================================
# Regression Metrics
# =====================================================

def regression_metrics(
    model,
    X_test,
    y_test
):
    """
    Calculate regression performance metrics.
    """

    predictions = model.predict(
        X_test
    )

    metrics = {

        "MAE":
            mean_absolute_error(
                y_test,
                predictions
            ),

        "MSE":
            mean_squared_error(
                y_test,
                predictions
            ),

        "RMSE":
            mean_squared_error(
                y_test,
                predictions
            ) ** 0.5,

        "R² Score":
            r2_score(
                y_test,
                predictions
            )
    }

    return metrics