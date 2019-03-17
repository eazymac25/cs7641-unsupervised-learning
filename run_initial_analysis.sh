#!/usr/bin/env bash

python run_experiment.py --benchmark --dataset1 --verbose --threads -1
python run_experiment.py --benchmark --dataset2 --verbose --threads -1

python run_experiment.py --ica --dataset1 --verbose --threads -1
python run_experiment.py --ica --dataset2 --verbose --threads -1

python run_experiment.py --pca --dataset1 --verbose --threads -1
python run_experiment.py --pca --dataset2 --verbose --threads -1

python run_experiment.py --rp  --dataset1 --verbose --threads -1
python run_experiment.py --rp  --dataset2 --verbose --threads -1

python run_experiment.py --rf  --dataset1 --verbose --threads -1
python run_experiment.py --rf  --dataset2 --verbose --threads -1
