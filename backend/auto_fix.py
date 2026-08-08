import pandas as pd


def auto_fix_dataset(df):

    df = df.copy()

    log = []

    # =====================================
    # Remove duplicates
    # =====================================

    duplicates = df.duplicated().sum()

    if duplicates:

        df = df.drop_duplicates()

        log.append(
            f"Removed {duplicates} duplicate rows."
        )

    # =====================================
    # Trim whitespace
    # =====================================

    object_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in object_cols:

        df[col] = df[col].astype(str).str.strip()

    log.append(
        "Trimmed whitespace."
    )

    # =====================================
    # Missing values
    # =====================================

    for col in df.columns:

        if df[col].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(
                df[col].median()
            )

        else:

            mode = df[col].mode()

            if not mode.empty:

                df[col] = df[col].fillna(
                    mode.iloc[0]
                )

    log.append(
        "Filled missing values."
    )

    # =====================================
    # Remove constant columns
    # =====================================

    constant = [

        c

        for c in df.columns

        if df[c].nunique() <= 1

    ]

    if constant:

        df = df.drop(columns=constant)

        log.append(
            f"Removed constant columns: {', '.join(constant)}"
        )

    # =====================================
    # Convert date columns
    # =====================================

    for col in df.columns:

        if df[col].dtype == object:

            try:

                converted = pd.to_datetime(
                    df[col],
                    errors="raise"
                )

                if converted.notna().sum() > len(df) * 0.8:

                    df[col] = converted

                    log.append(
                        f"Converted {col} to datetime."
                    )

            except Exception:

                pass

    return df, log