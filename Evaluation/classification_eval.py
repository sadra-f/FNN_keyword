import numpy as np
def binary_classification_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, and F1 score for binary classification.

    Args:
        y_true (array-like): Ground truth labels (0 or 1)
        y_pred (array-like): Predicted labels (0 or 1)

    Returns:
        dict: Dictionary with accuracy, precision, recall, and f1_score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Metrics
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
