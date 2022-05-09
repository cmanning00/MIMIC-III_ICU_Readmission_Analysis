# MIMIC-III_ICU_Readmission_Analysis
This is the source code for the paper 'Analysis and Prediction of Unplanned Intensive Care Unit Readmission using Recurrent Neural Networks with Long Short-Term Memory'
*[bioRxiv](https://www.biorxiv.org/content/early/2018/08/06/385518)

### Prerequisites

Please follow the original git files from MIMIC-III Benchmark Testing Codes.

```
git clone https://github.com/YerevaNN/mimic3-benchmarks
```

### Step-by-Step
Please follow the steps to get the results:

```
1. python3 scripts/extract_subjects.py [PATH TO MIMIC-III CSVs] data/root/
2. python3 scripts/validate_events.py data/root/
3. python3 scripts/create_readmission.py data/root/
4. python3 scripts/extract_episodes_from_subjects.py data/root/
5. python3 scripts/split_train_and_test.py data/root/
6. python3 scripts/create_readmission_data.py data/root/ data/readmission/
7. python3 mimic3models/split_train_val.py readmission_with_icustay_los
8. cd mimic3models/readmission3/
9. python -u main.py --network ../common_keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation
If you use this code for your research, please cite our [paper](https://www.biorxiv.org/content/early/2018/08/06/385518/):

```
@article{lin2018analysis,
  title={Analysis and Prediction of Unplanned Intensive Care Unit Readmission using Recurrent Neural Networks with Long Short-Term Memory},
  author={Lin, Yu-Wei and Zhou, Yuqian and Faghri, Faraz and Shaw, Michael J and Campbell, Roy H},
  journal={bioRxiv},
  pages={385518},
  year={2018},
  publisher={Cold Spring Harbor Laboratory}
}

```

>📋  A template README.md for code accompanying a Machine Learning paper

# Deep Learning for Healthcare Final Project

This repository is part of a project attempting to reimplement results from [Design and implementation of a deep recurrent model for prediction of readmission in urgent care using electronic health records](https://ieeexplore.ieee.org/document/8791466). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

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
