import pandas as pd


def dataset_health(df: pd.DataFrame):

    rows, cols = df.shape

    missing = df.isnull().sum().sum()

    duplicates = df.duplicated().sum()

    numeric = len(df.select_dtypes(include="number").columns)

    categorical = len(df.select_dtypes(exclude="number").columns)

    score = 100

    score -= min(missing, 30)

    score -= duplicates * 2

    score = max(score, 0)

    if score >= 90:
        status = "🟢 Excellent"

    elif score >= 75:
        status = "🟡 Good"

    elif score >= 60:
        status = "🟠 Fair"

    else:
        status = "🔴 Poor"

    return {

        "rows": rows,

        "columns": cols,

        "missing": missing,

        "duplicates": duplicates,

        "numeric": numeric,

        "categorical": categorical,

        "score": score,

        "status": status
    }