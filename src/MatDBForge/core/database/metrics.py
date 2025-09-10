"""Contains functions used to compute completeness metrics on the database."""

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel

from MatDBForge.active_learning.active_learning_utils import get_species_from_database


def vendi_score_bigO_n3(dataset: list[Atoms]):
    """Computes the Vendi Score for the database."""
    # Get dataset of length n
    len_dataset = len(dataset)

    # Getting the species from the database
    species = get_species_from_database(dataset)

    # Compute similarity matrix
    desc = SOAP(
        species=species,
        r_cut=6.0,
        n_max=8,
        l_max=6,
        sigma=0.2,
        # average='inner',
        periodic=True,
        compression={'mode': 'off'},
        sparse=False,
    )
    dataset_features = [desc.create(s) for s in dataset]
    # dataset_features = np.vstack(dataset_features)
    # breakpoint()
    re = AverageKernel(metric='rbf', gamma=1.0)
    sim_matrix = re.create(dataset_features)

    # Normalize similarity matrix by dividing by n
    normalized_matrix = sim_matrix / len_dataset

    # Compute eigenvalues of normalized similarity matrix
    eigenvalues = np.linalg.eigvalsh(normalized_matrix)

    # Compute Shannon entropy of eigenvalues
    # S_shannon = -\sum_{i=1}^n​ \lambda_i​ log(\lambda_i​)
    # here, we force the convention 0 log(0) = 0 by adding
    # a small epsilon
    eps = 1e-20
    shannon_entropy = -np.sum(eigenvalues * np.log(eigenvalues + eps))

    # Compute Vendi Score
    # VS=\exp(S_shannon)
    vendi_score = np.exp(shannon_entropy)

    return vendi_score


def vendi_score_lower_time_complexity(dataset: list[Atoms]):
    """Computes the Vendi Score for a dataset of n structures.

    This implementation has a lower time complexity than the
    O(n^3) implementation above, but requires that the embeddings
    are available, and that the embedding dimension d is
    significantly smaller than n (d << n). This is usually the case
    when using MLIPs, and for large datasets.
    """
    # Compute similarity matrix
    desc = SOAP(
        species=[29, 30],
        r_cut=6.0,
        n_max=8,
        l_max=6,
        sigma=0.2,
        # average='inner',
        periodic=True,
        compression={'mode': 'off'},
        sparse=False,
    )
    feature_matrix_X = np.vstack([desc.create(s) for s in dataset])

    # Stack the list of feature vectors into a single (n, d) numpy array
    # n = number of samples
    # d = dimension of each feature vector
    n, d = feature_matrix_X.shape

    # Compute the (d, d) covariance-like matrix C = X^T * X
    # This replaces the creation of the n x n similarity matrix
    # The complexity is then O(d^2 * n), instead of O(n^3), which comes
    # from computing the eigenvalues of the n x n similarity matrix.
    covariance_matrix_C = feature_matrix_X.T @ feature_matrix_X

    # Normalize similarity matrix by dividing by n
    normalized_matrix = covariance_matrix_C / n

    # Compute eigenvalues from the smaller matrix
    # The eigenvalues of K/n are the same as the eigenvalues of (X^T * X) / n
    # where K = X * X^T is the linear kernel matrix.
    eigenvalues = np.linalg.eigvalsh(normalized_matrix)
    print('eigenvalues: ', eigenvalues)

    # Compute Shannon entropy of eigenvalues
    # S_shannon = -\sum_{i=1}^n​ \lambda_i​ log(\lambda_i​)
    # Here, we force the convention 0 log(0) = 0 by adding a small epsilon.
    eps = 1e-20
    shannon_entropy = -np.sum(eigenvalues * np.log(eigenvalues + eps))

    # Compute Vendi Score
    # VS=\exp(S_shannon)
    vendi_score = np.exp(shannon_entropy)

    return vendi_score


def circles_metric():
    """Computes the Circles Metric for the database."""
    pass
