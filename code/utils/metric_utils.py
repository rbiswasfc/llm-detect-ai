from sklearn.metrics import roc_auc_score


def compute_metrics(predictions, truths):
    """
    ROC AUC SCORE
    """

    assert len(predictions) == len(truths)
    score = roc_auc_score(truths, predictions)

    to_return = {
        "lb": round(score, 4),
    }

    return to_return
