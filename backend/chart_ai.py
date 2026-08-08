import matplotlib.pyplot as plt
import pandas as pd

from backend.chart_planner import plan_chart


def generate_chart(df, question):

    plan = plan_chart(df, question)

    chart = plan.get("chart", "").lower()

    try:

        # =====================================================
        # Histogram
        # =====================================================

        if chart == "histogram":

            column = plan["x"]

            if column not in df.columns:
                return None

            fig, ax = plt.subplots(figsize=(8, 5))

            df[column].dropna().hist(ax=ax)

            ax.set_title(f"Histogram of {column}")

            return fig

        # =====================================================
        # Scatter Plot
        # =====================================================

        elif chart == "scatter":

            x = plan["x"]
            y = plan["y"]

            if x not in df.columns or y not in df.columns:
                return None

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.scatter(df[x], df[y])

            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{y} vs {x}")

            return fig

        # =====================================================
        # Line Chart
        # =====================================================

        elif chart == "line":

            x = plan["x"]
            y = plan["y"]

            if x not in df.columns or y not in df.columns:
                return None

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(df[x], df[y])

            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{y} vs {x}")

            return fig

        # =====================================================
        # Bar Chart
        # =====================================================

        elif chart == "bar":

            x = plan["x"]
            y = plan["y"]

            if x not in df.columns or y not in df.columns:
                return None

            fig, ax = plt.subplots(figsize=(9, 5))

            df.groupby(x)[y].mean().plot(
                kind="bar",
                ax=ax
            )

            ax.set_title(f"Average {y} by {x}")

            return fig

        # =====================================================
        # Box Plot
        # =====================================================

        elif chart == "box":

            column = plan["x"]

            if column not in df.columns:
                return None

            fig, ax = plt.subplots(figsize=(8, 5))

            df.boxplot(column=column, ax=ax)

            ax.set_title(f"Boxplot of {column}")

            return fig

        # =====================================================
        # Pie Chart
        # =====================================================

        elif chart == "pie":

            column = plan["x"]

            if column not in df.columns:
                return None

            fig, ax = plt.subplots(figsize=(7, 7))

            df[column].value_counts().plot(
                kind="pie",
                autopct="%1.1f%%",
                ax=ax
            )

            ax.set_ylabel("")

            return fig

        # =====================================================
        # Correlation Heatmap
        # =====================================================

        elif chart == "heatmap":

            numeric = df.select_dtypes(include="number")

            if numeric.shape[1] < 2:
                return None

            corr = numeric.corr()

            fig, ax = plt.subplots(figsize=(8, 6))

            c = ax.imshow(corr)

            plt.colorbar(c)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)

            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns)

            ax.set_title("Correlation Heatmap")

            return fig

        return None

    except Exception:

        return None