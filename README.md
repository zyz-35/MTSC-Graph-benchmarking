# Benchmarking Graph Representations and Graph Neural Networks for Multivariate Time Series Classification
The code for the paper: Benchmarking Graph Representations and Graph Neural Networks for Multivariate Time Series Classification

## Usage

### Training Command
```shell
python main.py --log_path path/to/save/logs --conf_file ./config/conf/raw-complete-megat-5-FingerMovements.yml --save_path path/to/save/model --gpu 0
```

## Data Preparation
Download all of the new 30 multivariate UEA Time Series Classification datasets <https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip

## Environment

### OS
Ubuntu 20.04.5, Ubuntu 20.04.1, Ubuntu 18.04.3, Ubuntu 16.04.4

### Required Package
- numpy
- pytorch
- sktime
- tqdm
- yaml
- yacs
