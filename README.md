# Bandit Two Clusters

A comprehensive implementation of bandit-based clustering algorithms for two-cluster problems.

## Overview

This repository implements and compares several clustering algorithms designed for scenarios where:
- Data points belong to one of two clusters
- Features can be sampled individually with associated costs
- The goal is to identify cluster assignments with minimal sampling budget

## Algorithms Implemented

### Core Algorithms
- **Gaussian Clustering**: Algorithms assuming Gaussian noise
- **Bernoulli Clustering**: Algorithms for Bernoulli-distributed responses
- **Bernoulli Sequential**: Sequential sampling version of Bernoulli clustering
- **K-means with Budget**: Budget-constrained K-means clustering

### Algorithm Components
Each clustering algorithm consists of three main components:
- **CESH** (Comparing Entries by Sequential Halving): Feature selection
- **CR** (Candidate Row): Identifies items from different clusters  
- **CBC** (Clustering by Candidate): Full clustering given a candidate

## Repository Structure

```
src/
├── algorithms/          # Core clustering algorithms
│   ├── gaussian.py     # Gaussian noise algorithms
│   ├── bernoulli.py    # Bernoulli algorithms
│   ├── bernoulli_sequential.py  # Sequential Bernoulli algorithms
│   └── kmeans.py       # Budget-constrained K-means
├── configs/            # Experiment configurations
│   ├── config_1.py     # Experiment 1: varying sparsity
│   ├── config_2.py     # Experiment 2: varying signal strength
│   ├── config_3.py     # Experiment 3: varying cluster proportions
│   └── config_4.py     # Experiment 4: compare to existing algorithm  
├── experiments/        # Experiment implementations
│   ├── experiment_1.py # Sparsity experiments
│   ├── experiment_2.py # Signal strength experiments
│   ├── experiment_3.py # Cluster proportion experiments
│   └── experiment_4.py # Adaptive sampling comparison
├── analysis/           # Results analysis and plotting
│   ├── results_analysis_1.py
│   ├── results_analysis_2.py
│   ├── results_analysis_3.py
│   └── results_analysis_4.py
├── utils/              # Utility functions
│   ├── data_generation.py  # Synthetic data generation
│   └── clustering.py       # Clustering evaluation utilities
├── runners/            # Experiment execution scripts
└── results/            # Generated results and plots
```

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn pandas
```

### Running Experiments

1. **Run a single experiment:**
```python
from src.experiments.experiment_1 import simulation_iteration_1
from src.configs.config_1 import config_1

# Run one iteration with random seed
results = simulation_iteration_1(seed=42)
```

2. **Run full Monte Carlo experiments:**
```python
from src.runners.run_experiment_1 import run_experiment1
run_experiment1()
```

3. **Analyze results:**
```python
# Results are automatically saved to src/results/
# Run analysis scripts to generate plots
exec(open('src/analysis/results_analysis_1.py').read())
```

## Experiments

### Experiment 1: Varying Sparsity
- **Purpose**: Study algorithm performance as feature sparsity changes
- **Fixed**: Signal strength, cluster proportions
- **Varied**: Number of informative features

### Experiment 2: Varying Signal Strength  
- **Purpose**: Study algorithm performance as matrix size
- **Fixed**: Sparsity, cluster proportions
- **Varied**: matrix size

### Experiment 3: Varying Cluster Proportions
- **Purpose**: Study algorithm performance with imbalanced clusters
- **Fixed**: Signal strength, sparsity
- **Varied**: Proportion of items in each cluster

### Experiment 4: Adaptive vs Bandit Comparison
- **Purpose**: Compare our algorithms against baselines
- **Focus**: Direct comparison of sample efficiency

## Key Features

- **Modular Design**: Each algorithm component is independently implementable
- **Reproducible Research**: Fixed random seeds and detailed configuration management
- **Efficient Implementation**: Optimized for large-scale Monte Carlo simulations

## Algorithm Details

### Input Format
All algorithms expect a `data_matrix` of shape `(num_items, num_features)` where:
- Rows represent items to be clustered
- Columns represent features that can be sampled
- Values represent the expected response for sampling that item-feature pair

### Output Format
Clustering algorithms return:
- `predicted_clusters`: Array of cluster assignments (0 or 1)
- `sample_cost`: Total number of samples used

### Configuration Parameters
- `delta`: Risk parameter (lower values = higher confidence)
- `budget`: Maximum allowed samples
- `monte_carlo_runs`: Number of simulation repetitions

## Performance Metrics

- **Sample Complexity**: Total samples required for clustering
- **Success Rate**: Fraction of experiments with perfect clustering



## Contributing

Please see [STYLE_GUIDE.md](STYLE_GUIDE.md) for coding standards and contribution guidelines.
