import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> plt.Figure:
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(num_classes)
    ax.set(xticks=ticks, yticks=ticks, xlabel="Predicted", ylabel="True", title="Confusion Matrix (test)")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.close(fig)
    return fig
