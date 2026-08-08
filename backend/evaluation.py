import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# =====================================================
# Confusion Matrix
# =====================================================

def plot_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(
        ax=ax,
        colorbar=False
    )

    plt.tight_layout()

    return fig


# =====================================================
# Classification Report
# =====================================================

def get_classification_report(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    return report


# =====================================================
# Feature Importance
# =====================================================

def get_feature_importance(
    model,
    feature_names
):

    if not hasattr(model, "feature_importances_"):
        return None

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.reset_index(
        drop=True,
        inplace=True
    )

    return importance