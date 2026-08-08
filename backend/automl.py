import pandas as pd

from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC


def run_automl(
    problem,
    X_train,
    X_test,
    y_train,
    y_test
):

    models = {}

    if problem == "Classification":

        models = {

            "Logistic Regression":
                LogisticRegression(max_iter=1000),

            "Decision Tree":
                DecisionTreeClassifier(),

            "Random Forest":
                RandomForestClassifier(),

            "KNN":
                KNeighborsClassifier(),

            "Support Vector Machine":
                SVC()

        }

    else:

        models = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(),

            "Random Forest":
                RandomForestRegressor()

        }

    scores = []

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        if problem == "Classification":

            score = accuracy_score(
                y_test,
                predictions
            )

        else:

            score = r2_score(
                y_test,
                predictions
            )

        scores.append(
            {
                "Model": name,
                "Score": score
            }
        )

    leaderboard = pd.DataFrame(scores)

    leaderboard = leaderboard.sort_values(
        by="Score",
        ascending=False
    )

    leaderboard.reset_index(
        drop=True,
        inplace=True
    )

    return leaderboard