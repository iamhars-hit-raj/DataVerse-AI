import streamlit as st


def show_model_comparison(leaderboard):

    st.divider()

    st.header("🏆 Model Comparison Dashboard")

    st.caption(
        "Visual comparison of all trained models."
    )

    # -----------------------------
    # Best Model Card
    # -----------------------------

    best = leaderboard.iloc[0]

    st.success(
        f"🥇 Best Model: **{best['Model']}** "
        f"({best['Score']:.4f})"
    )

    # -----------------------------
    # Progress Bars
    # -----------------------------

    st.subheader("Performance Ranking")

    max_score = leaderboard["Score"].max()

    for _, row in leaderboard.iterrows():

        score = row["Score"]

        progress = score / max_score if max_score != 0 else 0

        st.write(
            f"**{row['Model']}** — {score:.4f}"
        )

        st.progress(float(progress))

    # -----------------------------
    # Table
    # -----------------------------

    st.subheader("Leaderboard")

    st.dataframe(
        leaderboard,
        use_container_width=True
    )

    # -----------------------------
    # Chart
    # -----------------------------

    st.subheader("Model Scores")

    chart = leaderboard.set_index("Model")

    st.bar_chart(chart)