import streamlit as st


def navigation():

    st.sidebar.title("📊 DataVerse AI")

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📂 Upload Dataset",
            "📈 Dataset Overview",
            "🧹 Data Cleaning",
            "📊 Exploratory Data Analysis",
            "🧪 Feature Engineering",
            "⚙️ Machine Learning",
            "📄 Reports",
            "ℹ️ About"
        ]
    )

    return page