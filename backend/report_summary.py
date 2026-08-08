from datetime import datetime

from backend.ai_insights import generate_ai_insights


def build_report_summary(df, metrics=None):

    report = {}

    # =====================================================
    # Dataset Information
    # =====================================================

    report["rows"] = df.shape[0]
    report["columns"] = df.shape[1]
    report["missing"] = int(df.isnull().sum().sum())
    report["duplicates"] = int(df.duplicated().sum())

    report["column_names"] = list(df.columns)

    report["generated_on"] = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    # =====================================================
    # Model Metrics
    # =====================================================

    if metrics is None:

        report["metrics"] = {}

    else:

        report["metrics"] = metrics

    # =====================================================
    # AI Executive Insights
    # =====================================================

    try:

        report["insights"] = generate_ai_insights(df)

    except Exception:

        report["insights"] = (
            "AI Insights could not be generated."
        )

    return report