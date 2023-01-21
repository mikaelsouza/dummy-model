import os
import sys

import pandas as pd
import yaml
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

# Load params
params = yaml.safe_load(open("params.yaml"))["train"]
seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]

# Load args
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py input_path output_path\n")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Load data
train = pd.read_parquet(os.path.join(input_path, "train.parquet"))
test = pd.read_parquet(os.path.join(input_path, "test.parquet"))

# Extract X and y.
label_column = "Survived"

X_train = train.drop(label_column, axis=1)
y_train = train[label_column]

X_test = test.drop(label_column, axis=1)
y_test = test[label_column]

# Train model
model = RandomForestClassifier(
    n_estimators=n_est,
    min_samples_split=min_split,
    random_state=seed,
    n_jobs=2,
)
model.fit(X_train, y_train)

# Save model
if not os.path.exists(output_path):
    os.mkdir(output_path)
dump(model, os.path.join(output_path, "model.joblib"), compress=3)
