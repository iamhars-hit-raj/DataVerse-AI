from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
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
def select_features(df, target, features):

    if not features:
        return df

    selected_columns = features + [target]

    return df[selected_columns].copy()
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