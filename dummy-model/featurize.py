import os
import random
import sys

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# Loading data from params
params = yaml.safe_load(open("params.yaml"))["featurize"]
random.seed(params["seed"])

# Settings paths from args
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py input_file.csv output_path\n")
    sys.exit(1)

input_filepath = sys.argv[1]
input_train = os.path.join(input_filepath, "train.parquet")
input_test = os.path.join(input_filepath, "test.parquet")

output_path = sys.argv[2]
output_train = os.path.join(output_path, "train.parquet")
output_test = os.path.join(output_path, "test.parquet")

train_df = pd.read_parquet(input_train)
test_df = pd.read_parquet(input_test)

# Settings up features
numerical_features = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

# Setting up pipeline
pipeline = ColumnTransformer(
    [
        ("categorical", OrdinalEncoder(), categorical_features),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
pipeline.set_output(transform="pandas")

# Running pipeline
train_pipe_df = pipeline.fit_transform(train_df)
test_pipe_df = pipeline.transform(test_df)

# Saving featurized data
if not os.path.exists(output_path):
    os.mkdir(output_path)
train_pipe_df.to_parquet(output_train)
test_pipe_df.to_parquet(output_test)
