import streamlit as st

from frontend.hero import hero_section
from frontend.cards import metric_card
from frontend.data_health import show_health_score

def show_dashboard(df=None, filename=None):
    """
    Main Dashboard Page
    """

    # =====================================================
    # Hero Section
    # =====================================================
    hero_section()

    # =====================================================
    # Check Dataset
    # =====================================================
    if df is None:
        st.info("👈 Upload a dataset from the sidebar to begin.")
        return

    # =====================================================
    # Dataset Statistics
    # =====================================================
    rows = df.shape[0]
    cols = df.shape[1]
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    # =====================================================
    # KPI Cards
    # =====================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "📄",
            "Total Rows",
            f"{rows:,}",
            "Dataset Records"
        )

    with col2:
        metric_card(
            "📊",
            "Columns",
            cols,
            "Available Features"
        )

    with col3:
        metric_card(
            "❌",
            "Missing Values",
            missing,
            "Null Entries"
        )

    with col4:
        metric_card(
            "📦",
            "Duplicate Rows",
            duplicates,
            "Repeated Records"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # Dataset Health
    # =====================================================

    show_health_score(df)
    st.divider()

    # =====================================================
    # Active Dataset
    # =====================================================
    st.subheader("📁 Active Dataset")

    st.success(filename)

    st.divider()

    # =====================================================
    # Quick Actions
    # =====================================================
    st.subheader("⚡ Quick Actions")

    action1, action2, action3 = st.columns(3)

    with action1:
        st.button(
            "🧹 Data Cleaning",
            use_container_width=True
        )

    with action2:
        st.button(
            "📊 Exploratory Data Analysis",
            use_container_width=True
        )

    with action3:
        st.button(
            "🤖 Machine Learning",
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # Project Status
    # =====================================================
    st.subheader("🚀 Development Roadmap")

    st.progress(0.35)

    st.markdown("## 🚀 Development Roadmap")

    progress = 60
    st.progress(progress)

    st.markdown("### ✅ Completed")

    completed = [
        "Dataset Upload",
        "Session State Management",
        "Dataset Overview",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Dashboard",
        "KPI Cards",
        "Dataset Health Score",
        "Dashboard Charts",
        "Feature Engineering",
        "Machine Learning Studio",
        "AI Analyst",
        "AI Dashboard",
        "AI Copilot",
        "Reports",
    ]

    for item in completed:
        st.markdown(f"- {item}")

    st.markdown("### 🚧 In Progress")

    in_progress = [
        "Advanced Machine Learning",
        "Advanced Model Evaluation",
        "AutoML",
        "Explainable AI (SHAP)",
    ]

    for item in in_progress:
        st.markdown(f"- {item}")

    st.markdown("### 🔜 Coming Soon")

    coming_soon = [
        "AI Dataset Assistant",
        "Automated Data Storytelling",
        "Model Deployment / Prediction API",
        "Advanced Feature Selection",
    ]

    for item in coming_soon:
        st.markdown(f"- {item}")