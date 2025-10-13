from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score,precision_score,recall_score
import sys
from networksecurity.exception.exceptions import NetworkSecurityError


def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    """
    Calculates classification metrics: F1 score, precision score, and recall score.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        ClassificationMetricArtifact: An object containing the calculated metrics.
    """
    try:
        f1=f1_score(y_true,y_pred)
        precision=precision_score(y_true,y_pred)
        recall=recall_score(y_true,y_pred)
        classification_metric_artifact=ClassificationMetricArtifact(f1_score=f1,precision_score=precision,recall_score=recall)
        return classification_metric_artifact
    except Exception as e:
        raise NetworkSecurityError(e, sys) from e