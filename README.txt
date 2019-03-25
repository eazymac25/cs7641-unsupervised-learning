# CS 7641-unsupervised-learning
Georgia Tech CS 7641 - Unsupervised Learning project

Author: Kyle MacNeney

Author Email: kyle.macneney@gmail.com

Repository: https://github.com/eazymac25/cs7641-unsupervised-learning

**NOTE: Written in markdown**

Happily stolen from https://github.com/cmaron/CS-7641-assignments/tree/master/assignment3

## Description

### Pre-requisites
1. Install Python 3.5 (preferably via [Anaconda](https://www.anaconda.com/))
2. Install git

### Installation Instructions
1. Clone or download
```bash
git clone https://github.com/eazymac25/cs7641-unsupervised-learning.git
```
2. Install all requirements from `requirements.txt`
```bash
pip install -r requirements.txt
```

### Directory Structure
```
|-- run_experiment.py
|-- run_clustering.sh (run after choosing dimension from scree plots)
|-- run_initial_analysis.sh (run to generate inital scree plots)
|-- README.md
|-- README.txt (because you asked for it)
|-- README-original.md (where this came from)
| -- data
    |-- loader.py (transforms data into pandas.DataFrame)
    |-- original_loader.py (dependency for loader.py)
    |-- raw_census_data.csv (census dataset)
    |-- winequality-red.csv (wine dataset)
|-- experiments
    |-- base.py
    |-- benchmark.py (benchmark experiment)
    |-- clustering.py
    |-- ICA.py
    |-- PCA.py
    |-- plotting.py
    |-- RF.py (random forest dr experiment)
    |-- RP.py
    |-- other unused files
|-- output
    |-- PCA
    |-- ICA
    |-- benchmark
    |-- RP
    |-- RF
    |-- images (Graphs from experiments
```

### Overview of the Pipeline

In order to operate the experiments, we must:

1. Run the benchmark clustering without DR (K-Means and Expectation Maximization) and run the original
neural network using the clusters
2. Run the Dimensionality Reduction
3. Run Dimensionality Reduction with a specific dimension, run clustering, and run the neural net.

To keep all things equal, we are using the same features used in the previous experiment.
While This may seem silly, it is done to reduce complexity.
Further analysis should be done with other feature extraction techniques but the same pipeline.

### Running Experiments
First we must benchmark and run dimensionality reduction to generate scree plots
**NOTE: Dataset1 = Census Dataset2=Wine (see `run_experiment.py`)**
```bash
python run_experiment.py --all
```
Or Run individually like:
This runs the benchmark
```bash
python run_experiment.py --benchmark --dataset1 --verbose --threads 1
```

This runs a specific DR algo on the wine dataset
```bash
python run_experiment.py --pca --dataset2 --verbose --threads 1
```

Once a dimension has been established for each variable update `run_clustering.sh`
with that dimension for each case and then run
```bash
./run_clustering
```

Or run each clustering experiment individually

```bash
python run_experiment.py --pca --dataset1 --dim 5 --skiprerun --verbose --threads 1
```
This skips the initial rerun to generate screen and
1. Runs DR
2. Passes DR through NN
3. Saves DR results
4. Runs experiment with DR results passes DR results through both (K-Means and EM)
5. Runs output of DR -> Clustering through NN

### Plotting

At any time you may run

```bash
python run_experiment.py --plot
```

This will plot all the data generated so far.