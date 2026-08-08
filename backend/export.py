import joblib
import tempfile
import pandas as pd


def save_model(model):

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pkl"
    )

    joblib.dump(
        model,
        temp.name
    )

    return temp.name


def prediction_dataframe(
    X_test,
    y_test,
    predictions
):

    df = X_test.copy()

    df["Actual"] = y_test.values

    df["Predicted"] = predictions

    return df