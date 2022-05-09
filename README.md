# Deep Learning for Healthcare Final Project

This repository is part of a project attempting to reimplement results from [Design and implementation of a deep recurrent model for prediction of readmission in urgent care using electronic health records](https://ieeexplore.ieee.org/document/8791466). 

## Requirements

### Download
1.  Data: before using this code, you must
2.  You will need the following packages:
- pandas
- numpy
- keras
- scikit-learn
- frozendict
- matplotlib

3.  Download the file [claims_codes_hst_300.txt](https://github.com/clinicalml/embeddings).  Add this file to the `mimic3-readmission/embeddings` folder

4.  Type the command below in bash:
- export PYTHONPATH=$PYTHONPATH:[PATH TO THIS REPOSITORY]

##MIMIC-III Benchmarks - Data Preprocessing [source](https://github.com/YerevaNN/mimic3-benchmarks)
Run the following commands:
1.  python scripts/extract_subjects.py [MIMIC-III CSVs PATH] [OUTPUT PATH]
- Produces various files on data subjects such as `stays.csv`, `events.csv`, and `diagnoses.csv`
3.  python scripts/validate_events.py [OUTPUT PATH]
- Fixes some problems with mising data
5.  python scripts/create_readmission.py [OUTPUT PATH]
- Produces the file `stays_readmission.csv` is the result of the patient class-labeling process
6.  python scripts/extract_episodes_from_subjects.py [OUTPUT PATH]
- Extracts time-series data
8.  python scripts/create_readmission_data.py [OUTPUT PATH] [OUTPUT PATH 2]
- Remove patients who died in the ICU and generate final preprocessed data
10.  python scripts/split_train_val_test.py
- Divides the data into a training, validation, and test sets

## Training

Use these command to train the baseline models:

```
cd /mimic3models/readmission_baselines/logistic_cv_0
python svm_s_p.py
```

Use these commands to train the LSTM+CNN model:

```
cd /mimic3models/readmission/
python3 -u main.py --network ../common_keras_models/lstm_cnn.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 
```

## Testing and Evaluation

Use these commands to train the LSTM+CNN model:

```
cd /mimic3models/readmission/
python3 -u main.py --network ../common_keras_models/lstm_cnn.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8
```


##References

[1] https://github.com/YerevaNN/mimic3-benchmarks

[2] https://github.com/clinicalml/embeddings
