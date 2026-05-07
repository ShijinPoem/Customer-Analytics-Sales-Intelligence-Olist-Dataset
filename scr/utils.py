import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix"
):
    """
    Plot confusion matrix heatmap.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


def plot_feature_importance(
    importance_df,
    title="Feature Importance"
):
    """
    Plot feature importance chart.
    """

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance_df,
        x='importance',
        y='feature'
    )

    plt.title(title)

    plt.show()