import shap
import matplotlib.pyplot as plt


def shap_summary_plot(model, X_train):
    """
    Generates a SHAP summary plot for supported tree-based models.
    Returns a matplotlib Figure or None if unsupported.
    """

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        plt.close("all")

        fig = plt.figure(figsize=(10, 6))

        shap.summary_plot(
            shap_values,
            X_train,
            show=False
        )

        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"SHAP Error: {e}")
        return None