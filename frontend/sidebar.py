import streamlit as st


def navigation():

    # ==========================================
    # Logo
    # ==========================================

    st.sidebar.markdown("## 📊 DataVerse AI")

    st.sidebar.caption(
        "AI-Powered Data Science Platform"
    )

    st.sidebar.markdown("---")

    # ==========================================
    # Navigation
    # ==========================================

    page = st.sidebar.radio(
        "📌 Navigation",
        [
            "🏠 Dashboard",
            "📂 Upload Dataset",
            "📈 Dataset Overview",
            "🧹 Data Cleaning",
            "📊 Exploratory Data Analysis",
            "🧪 Feature Engineering",
            "⚙️ Machine Learning",
            "🤖 AI Analyst",
            "📈 AI Dashboard",
            "🧠 AI Copilot",
            "📄 Reports",
            "ℹ️ About"
        ],
        index=0
    )

    st.sidebar.markdown("---")

    # ==========================================
    # Platform Information
    # ==========================================

    st.sidebar.info(
        """
**DataVerse AI**

Version: **1.0**

Built with:

- 🐍 Python
- 🎈 Streamlit
- 🐼 Pandas
- 📊 Plotly
- 🤖 Scikit-Learn
"""
    )

    st.sidebar.markdown("---")

    st.sidebar.caption("© 2026 DataVerse AI")

    return page