import pandas as pd
import streamlit as st


def prediction_playground(model, X_train):

    st.divider()

    st.header("🎯 Prediction Playground")

    st.caption(
        "Enter feature values and let the trained model make a prediction."
    )

    user_input = {}

    for column in X_train.columns:

        dtype = X_train[column].dtype

        # Numeric Columns
        if pd.api.types.is_numeric_dtype(dtype):

            value = st.number_input(
                column,
                value=float(X_train[column].median())
            )

        # Categorical Columns
        else:

            options = list(
                X_train[column].dropna().unique()
            )

            value = st.selectbox(
                column,
                options
            )

        user_input[column] = value

    if st.button(
        "🚀 Predict",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [user_input]
        )

        prediction = model.predict(
            input_df
        )

        st.success(
            f"Prediction: {prediction[0]}"
        )

        # Show probabilities if available
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_df
            )[0]

            st.subheader(
                "Prediction Probabilities"
            )

            prob_df = pd.DataFrame(
                {
                    "Class": model.classes_,
                    "Probability": probabilities
                }
            )

            st.dataframe(
                prob_df,
                use_container_width=True
            )