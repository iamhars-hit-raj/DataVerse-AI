import streamlit as st
import pandas as pd


def show_dataset_overview(df):

    rows, columns = df.shape
    total_missing = df.isnull().sum().sum()
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    st.subheader("📈 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", rows)

    with col2:
        st.metric("Columns", columns)

    with col3:
        st.metric("Missing Values", total_missing)

    with col4:
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.divider()

    st.subheader("📝 Column Names")

    st.write(", ".join(df.columns))

    st.divider()

    st.subheader("📊 Data Types")

    datatype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        datatype_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("❌ Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )