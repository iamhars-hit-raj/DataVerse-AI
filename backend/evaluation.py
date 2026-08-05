import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


def plot_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(ax=ax)

    return fig


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