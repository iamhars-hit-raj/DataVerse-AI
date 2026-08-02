import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.model_selection import train_test_split


# =====================================================
# Analyze Features
# =====================================================

def analyze_features(df):

    return {
        "numeric": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime"]).columns.tolist(),
        "boolean": df.select_dtypes(include=["bool"]).columns.tolist()
    }


# =====================================================
# Missing Value Imputation
# =====================================================

def impute_numeric(df, column, method):

    df = df.copy()

    if method == "Mean":
        df[column] = df[column].fillna(df[column].mean())

    elif method == "Median":
        df[column] = df[column].fillna(df[column].median())

    elif method == "Zero":
        df[column] = df[column].fillna(0)

    return df


def impute_categorical(df, column):

    df = df.copy()

    mode = df[column].mode()

    if not mode.empty:
        df[column] = df[column].fillna(mode.iloc[0])

    return df


# =====================================================
# Encoding
# =====================================================

def label_encode(df, column):

    df = df.copy()

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column].astype(str))

    return df


def one_hot_encode(df, column):

    df = pd.get_dummies(
        df,
        columns=[column],
        dtype=int
    )

    return df


# =====================================================
# Feature Scaling
# =====================================================

def scale_features(df, columns, method):

    df = df.copy()

    if not columns:
        return df

    if method == "StandardScaler":
        scaler = StandardScaler()

    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()

    else:
        scaler = RobustScaler()

    df[columns] = scaler.fit_transform(df[columns])

    return df


# =====================================================
# Feature Selection
# =====================================================

def select_features(df, target, features):

    if not features:
        return df.copy()

    selected_columns = features + [target]

    return df[selected_columns].copy()


# =====================================================
# Train/Test Split
# =====================================================

def split_dataset(df, target, train_size, random_state, shuffle):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test