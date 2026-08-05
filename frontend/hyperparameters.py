import streamlit as st


def get_hyperparameters(model_name):

    params = {}

    if model_name == "Random Forest":

        params["n_estimators"] = st.slider(
            "Number of Trees",
            10,
            500,
            100
        )

        params["max_depth"] = st.slider(
            "Max Depth",
            1,
            50,
            10
        )

        params["random_state"] = 42

    elif model_name == "Decision Tree":

        params["max_depth"] = st.slider(
            "Max Depth",
            1,
            50,
            10
        )

        params["min_samples_split"] = st.slider(
            "Min Samples Split",
            2,
            20,
            2
        )

    elif model_name == "KNN":

        params["n_neighbors"] = st.slider(
            "Neighbors",
            1,
            25,
            5
        )

    elif model_name == "Support Vector Machine":

        params["kernel"] = st.selectbox(
            "Kernel",
            ["rbf", "linear", "poly", "sigmoid"]
        )

        params["C"] = st.slider(
            "Regularization",
            0.1,
            10.0,
            1.0
        )

    elif model_name == "Logistic Regression":

        params["solver"] = st.selectbox(
            "Solver",
            ["lbfgs", "liblinear", "newton-cg"]
        )

        params["C"] = st.slider(
            "Regularization",
            0.1,
            10.0,
            1.0
        )

        params["max_iter"] = 1000

    return params