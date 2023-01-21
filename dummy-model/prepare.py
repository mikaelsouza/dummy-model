import os
import random
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Loading data from params
params = yaml.safe_load(open("params.yaml"))["prepare"]
test_size = params["test_size"]
random.seed(params["seed"])

# Settings paths from args
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py input_file.csv output_path\n")
    sys.exit(1)

input_filepath = sys.argv[1]

output_path = sys.argv[2]
output_train = os.path.join(output_path, "train.parquet")
output_test = os.path.join(output_path, "test.parquet")

# Preparing data
data = pd.read_csv(input_filepath)
dropped_columns = ["PassengerId", "Cabin", "Name", "Ticket"]
data = data.drop(dropped_columns, axis=1).dropna()
train, test = train_test_split(data, test_size=test_size)

# Saving data to file
if not os.path.exists(output_path):
    os.mkdir(output_path)
train.to_parquet(output_train)
test.to_parquet(output_test)
