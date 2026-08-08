import pandas as pd
import matplotlib.pyplot as plt

from backend.ai_insights import generate_ai_insights

import plotly.express as px
import plotly.figure_factory as ff

from backend.dashboard_planner import plan_dashboard
from frontend import dashboard

def generate_dashboard(df):

    dashboard = {}

    # ==========================================
    # KPI Cards
    # ==========================================

    dashboard["rows"] = df.shape[0]
    dashboard["columns"] = df.shape[1]
    dashboard["missing"] = int(df.isnull().sum().sum())
    dashboard["duplicates"] = int(df.duplicated().sum())

    # ==========================================
    # Correlation Heatmap
    # ==========================================

    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] >= 2:

        fig, ax = plt.subplots(figsize=(8, 6))

        corr = numeric.corr()

        c = ax.imshow(corr)

        plt.colorbar(c)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)

        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        ax.set_title("Correlation Heatmap")

        dashboard["heatmap"] = fig

    else:

        dashboard["heatmap"] = None

    # ==========================================
    # AI Dashboard Planner
    # ==========================================

    plan = plan_dashboard(df)

    dashboard["title"] = plan.get(
        "title",
        "AI Dashboard"
    )

    dashboard["summary"] = plan.get(
        "summary",
        ""
    )

    dashboard["recommendations"] = plan.get(
        "recommendations",
        []
    )

    dashboard["charts"] = []

    for chart in plan.get("charts", []):

        try:

            chart_type = chart.get(
                "type",
                ""
            ).lower()

            if chart_type == "bar":

                fig = px.bar(
                    df,
                    x=chart["x"],
                    y=chart["y"]
                )

            elif chart_type == "line":

                fig = px.line(
                    df,
                    x=chart["x"],
                    y=chart["y"]
                )

            elif chart_type == "scatter":

                fig = px.scatter(
                    df,
                    x=chart["x"],
                    y=chart["y"]
                )

            elif chart_type == "histogram":

                fig = px.histogram(
                    df,
                    x=chart["x"]
                )

            elif chart_type == "box":

                fig = px.box(
                    df,
                    y=chart["x"]
                )

            elif chart_type == "pie":

                counts = (
                    df[chart["x"]]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    chart["x"],
                    "Count"
                ]

                fig = px.pie(
                    counts,
                    names=chart["x"],
                    values="Count"
                )

            elif chart_type == "heatmap":

                numeric = df.select_dtypes(
                    include="number"
                )

                corr = numeric.corr()

                fig = ff.create_annotated_heatmap(
                    z=corr.values,
                    x=list(corr.columns),
                    y=list(corr.index),
                    annotation_text=round(
                        corr,
                        2
                    ).values,
                    showscale=True
                )

            else:

                continue

            dashboard["charts"].append(fig)

        except Exception:

            continue
    # ==========================================
    # AI Insights
    # ==========================================

    dashboard["insights"] = generate_ai_insights(df)

    return dashboard