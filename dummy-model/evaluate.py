import json
import math
import os
import sys

import pandas as pd
from dvclive import Live
from joblib import load
from sklearn import metrics

EVAL_PATH = "eval"


def evaluate(model, df: pd.DataFrame, split: str, label_col: str, live):
    # Split into features and labels
    X = df.drop(label_col, axis=1)
    y = df[label_col]

    # Run prediction
    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]
    prediction_labels = predictions_by_class.argmax(-1)

    # Logging scalar metrics
    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)

    if not live.summary:
        live.summary = {"avg_prec": {}, "roc_auc": {}}
    live.summary["avg_prec"][split] = avg_prec
    live.summary["roc_auc"][split] = roc_auc

    # Logging plots
    live.log_sklearn_plot("roc", y, predictions, name=f"roc/{split}")
    live.log_sklearn_plot("precision_recall", y, predictions, name=f"prc/{split}")
    live.log_sklearn_plot("confusion_matrix", y, prediction_labels, name=f"cm/{split}")


if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model_path data_path\n")
    sys.exit(1)

model_file = sys.argv[1]
train_file = os.path.join(sys.argv[2], "train.parquet")
test_file = os.path.join(sys.argv[2], "test.parquet")

with open(model_file, "rb") as f:
    model = load(f)

train = pd.read_parquet(train_file)
test = pd.read_parquet(test_file)

live = Live(os.path.join(EVAL_PATH, "live"))
evaluate(model, train, "train", "Survived", live)
evaluate(model, test, "test", "Survived", live)
live.make_summary()
