prepare:
	dvc stage add --force -n prepare \
				  -p prepare.seed,prepare.test_size \
				  -d dummy-model/prepare.py \
				  -d data/raw/titanic-dataset.csv \
				  -o data/prepared/ \
				  python dummy-model/prepare.py data/raw/titanic-dataset.csv data/prepared

featurize:
	dvc stage add --force -n featurize \
				  -p featurize.seed \
				  -d dummy-model/featurize.py \
				  -d data/prepared/train.parquet \
				  -d data/prepared/test.parquet \
				  -o data/featurized \
				  python dummy-model/featurize.py data/prepared data/featurized

train:
	dvc stage add --force -n train \
				  -p train.seed,train.n_est,train.min_split \
				  -d dummy-model/train.py \
				  -d data/featurized/train.parquet \
				  -d data/featurized/test.parquet \
				  -o models/artifacts/ \
				  python dummy-model/train.py data/featurized models/artifacts

evaluate:
	dvc stage add --force -n evaluate \
				  -d dummy-model/evaluate.py \
				  -d models/artifacts/model.joblib \
				  -d data/featurized/train.parquet \
				  -d data/featurized/test.parquet \
				  -M eval/live/metrics.json \
				  -O eval/live/plots \
				  python dummy-model/evaluate.py models/artifacts/model.joblib data/featurized/