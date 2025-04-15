import csv
import logging

import numpy as np # type: ignore


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_matrix(num_items, num_features, sparsity, signal_strength, true_clusters):
    """
    Create a matrix M based on the given parameters.

    Parameters:
    ----------
    num_items : int
        Number of rows (samples).
    num_features : int
        Number of columns (features).
    sparsity : int
        Number of non-zero elements in the feature vector.
    signal_strength : float
        Signal strength.
    true_clusters : list
        List of cluster labels (1 or 2) for each row.

    Returns:
    -------
    np.ndarray
        The generated matrix means_matrix.
    """
    m_a = np.zeros(num_features)
    m_b = np.zeros(num_features)
    m_b[:sparsity] = signal_strength / np.sqrt(sparsity)
    means_matrix = np.zeros((num_items, num_features))
    for i in range(num_items):
        means_matrix[i, :] = m_a if true_clusters[i] == 1 else m_b
    return means_matrix


def save_results(data: list, filename: str, headers: list = None):
    """
    Save experiment results to a CSV file.

    Parameters:
    ----------
    data : list or np.ndarray
        The data to save (2D array or list of lists).
    filename : str
        The name of the output CSV file.
    headers : list, optional
        List of column headers for the CSV file.
    """
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
    logging.info(f"Results saved to {filename}")


def load_results(filename: str) -> list:
    """
    Load experiment results from a CSV file.

    Parameters:
    ----------
    filename : str
        The name of the input CSV file.

    Returns:
    -------
    list
        The loaded data as a list of lists.
    """
    with open(filename, mode="r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    logging.info(f"Results loaded from {filename}")
    return data


def log_experiment_start(experiment_name: str, config: dict):
    """
    Log the start of an experiment with its configuration.

    Parameters:
    ----------
    experiment_name : str
        The name of the experiment.
    config : dict
        The configuration dictionary for the experiment.
    """
    logging.info(f"Starting experiment: {experiment_name}")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")


def log_experiment_end(experiment_name: str):
    """
    Log the end of an experiment.

    Parameters:
    ----------
    experiment_name : str
        The name of the experiment.
    """
    logging.info(f"Experiment {experiment_name} completed.")
