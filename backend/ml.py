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

def get_model(problem_type, model_name):

    if problem_type == "Classification":

        models = {

            "Logistic Regression": LogisticRegression(),

            "Decision Tree": DecisionTreeClassifier(),

            "Random Forest": RandomForestClassifier(),

            "KNN": KNeighborsClassifier(),

            "Support Vector Machine": SVC()

        }

    else:

        models = {

            "Linear Regression": LinearRegression(),

            "Decision Tree": DecisionTreeRegressor(),

            "Random Forest": RandomForestRegressor()

        }

    return models[model_name]


# =====================================================
# Train Model
# =====================================================

def train_model(

    model,

    X_train,

    y_train

):

    model.fit(

        X_train,

        y_train

    )

    return model


# =====================================================
# Classification Metrics
# =====================================================

def classification_metrics(

    model,

    X_test,

    y_test

):

    predictions = model.predict(X_test)

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

            average="weighted"

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

    predictions = model.predict(X_test)

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