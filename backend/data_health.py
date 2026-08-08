import pandas as pd


def dataset_health(df: pd.DataFrame):

    rows, cols = df.shape

    missing = df.isna().sum().sum()

    duplicates = df.duplicated().sum()

    numeric = len(
        df.select_dtypes(include="number").columns
    )

    categorical = len(
        df.select_dtypes(exclude="number").columns
    )

    constant_cols = [
        c for c in df.columns
        if df[c].nunique() == 1
    ]

    high_cardinality = []

    for col in df.select_dtypes(include="object").columns:

        if df[col].nunique() > len(df) * 0.5:

            high_cardinality.append(col)

    score = 100

    score -= min(20, missing)

    score -= min(10, duplicates)

    score -= len(constant_cols) * 2

    score -= len(high_cardinality)

    score = max(score, 0)

    if score >= 90:
        status = "🟢 Excellent"

    elif score >= 75:
        status = "🟡 Good"

    elif score >= 60:
        status = "🟠 Fair"

    else:
        status = "🔴 Poor"

    issues = []

    if missing:
        issues.append(f"Missing Values : {missing}")

    if duplicates:
        issues.append(f"Duplicate Rows : {duplicates}")

    if constant_cols:
        issues.append(
            f"Constant Columns : {len(constant_cols)}"
        )

    if high_cardinality:
        issues.append(
            f"High Cardinality Columns : {len(high_cardinality)}"
        )

    return {

        "rows": rows,

        "columns": cols,

        "numeric": numeric,

        "categorical": categorical,

        "missing": missing,

        "duplicates": duplicates,

        "score": score,

        "status": status,

        "issues": issues
    }