import streamlit as st
import pandas as pd

from backend.auto_fix import auto_fix_dataset


def data_cleaning(df):

    st.subheader("🧹 Data Cleaning")

    # ====================================
    # Duplicate Rows
    # ====================================

    duplicate_count = df.duplicated().sum()

    st.info(f"Duplicate Rows Found: {duplicate_count}")

    if st.button("🗑 Remove Duplicate Rows"):

        original_rows = len(df)

        df = df.drop_duplicates()

        removed = original_rows - len(df)

        st.session_state.df = df

        st.success(
            f"Successfully removed {removed} duplicate rows."
        )

    st.divider()

    # ====================================
    # Missing Values
    # ====================================

    st.subheader("❌ Handle Missing Values")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

    method = st.selectbox(
        "Choose Missing Value Strategy",
        [
            "Do Nothing",
            "Drop Rows",
            "Fill Mean",
            "Fill Median",
            "Fill Mode"
        ]
    )

    if st.button("Apply Missing Value Strategy"):

        if method == "Drop Rows":

            before = len(df)

            df = df.dropna()

            st.success(
                f"Removed {before-len(df)} rows containing missing values."
            )

        elif method == "Fill Mean":

            numeric = df.select_dtypes(include="number").columns

            for col in numeric:

                df[col] = df[col].fillna(
                    df[col].mean()
                )

            st.success(
                "Filled numeric missing values using Mean."
            )

        elif method == "Fill Median":

            numeric = df.select_dtypes(include="number").columns

            for col in numeric:

                df[col] = df[col].fillna(
                    df[col].median()
                )

            st.success(
                "Filled numeric missing values using Median."
            )

        elif method == "Fill Mode":

            for col in df.columns:

                mode = df[col].mode()

                if not mode.empty:

                    df[col] = df[col].fillna(
                        mode.iloc[0]
                    )

            st.success(
                "Filled missing values using Mode."
            )

        else:

            st.info(
                "No changes applied."
            )

        st.session_state.df = df

    st.divider()

    # ====================================
    # AI Auto Fix
    # ====================================

    st.subheader("✨ AI Auto Fix")

    st.caption(
        "Automatically clean the dataset using intelligent preprocessing."
    )

    if st.button(
        "🚀 Auto Fix Dataset",
        use_container_width=True
    ):

        with st.spinner(
            "Applying AI fixes..."
        ):

            cleaned_df, logs = auto_fix_dataset(df)

            df = cleaned_df

            st.session_state.df = df

        st.success(
            "✅ Dataset cleaned successfully!"
        )

        st.subheader("📝 Changes Applied")

        if logs:

            for item in logs:

                st.write("✅", item)

        else:

            st.info(
                "No changes were required."
            )

        st.dataframe(
            df.head(),
            use_container_width=True,
            hide_index=True
        )

    return df