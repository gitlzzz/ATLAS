"""Contains functions used to compute completeness metrics on the database."""

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry.analysis import get_rdf
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler, normalize

from MatDBForge.active_learning.active_learning_utils import (
    generate_descriptors,
    get_species_from_database,
)
from MatDBForge.core.code_utils import custom_print


def get_vendi_score_db(
    dataset: list[Atoms],
    descriptor_type: str = 'soap',
    descriptor_settings: dict = None,
    mace_model_path: str = None,
    use_custom_features: bool = True,
):
    """Computes the Vendi Score for a dataset of n structures.

    This implementation has a lower time complexity than the
    O(n^3) implementation above, but requires that the embeddings
    are available, and that the embedding dimension d is
    significantly smaller than n (d << n). This is usually the case
    when using MLIPs, and for large datasets.
    """
    custom_print(
        f'Getting feature matrix for {len(dataset)} '
        f'structures using {descriptor_type.upper()}...'
    )

    _, feature_matrix_X = generate_descriptors(
        database=dataset,
        descriptor_type=descriptor_type,
        descriptor_settings=descriptor_settings,
        model_path=mace_model_path,
    )

    # feature_matrix_X = np.vstack([desc.create(s) for s in dataset])

    if use_custom_features:
        custom_print('Adding custom features to feature matrix...')
        # Adding custom features to turn the local descriptors into pseudo-global ones
        # Adding number of atoms as a feature
        db_sizes = np.array([len(struct) for struct in dataset])
        feature_matrix_X = np.concatenate(
            (feature_matrix_X, db_sizes.reshape(-1, 1)), axis=1
        )

        # Adding species fractions as features
        species = get_species_from_database(dataset)
        species_frac = np.zeros((len(dataset), len(species)))
        species_numbers = []
        for i, struct in enumerate(dataset):
            species_frac[i] = [struct.symbols.count(s) / len(s) for s in species]
            species_numbers.append(
                [
                    atomic_numbers[s] if struct.symbols.count(s) > 0 else 0
                    for s in species
                ]
            )
        feature_matrix_X = np.concatenate((feature_matrix_X, species_frac), axis=1)

        # Adding species number as a feature
        species_numbers = np.array(species_numbers)
        feature_matrix_X = np.concatenate((feature_matrix_X, species_numbers), axis=1)

        # Adding cell volume as a feature
        db_cell_data = np.array(
            [
                [struct.get_volume()]
                + list(struct.cell.angles())
                + list(struct.cell.lengths())
                for struct in dataset
            ]
        )
        feature_matrix_X = np.concatenate((feature_matrix_X, db_cell_data), axis=1)

        # Adding atomic density as a feature
        density = np.array([len(struct) / struct.get_volume() for struct in dataset])
        feature_matrix_X = np.concatenate(
            (feature_matrix_X, density.reshape(-1, 1)), axis=1
        )

        # Get rdf of each structure and add as features
        new_feature_matrix = []
        for i, struct in enumerate(dataset):
            # Compute RDF using ASE's get_rdf function
            # Parameters can be adjusted as needed
            try:
                rdf = get_rdf(struct, rmax=10.0, nbins=50, no_dists=True)
            except Exception:
                rdf = get_rdf(struct, rmax=0.5, nbins=50, no_dists=True)

            # Append RDF to feature matrix
            new_feature_matrix.append(np.concatenate((feature_matrix_X[i], rdf)))
        new_feature_matrix = np.array(new_feature_matrix)

        # Standardizing features
        # This scales each feature (column) to have a mean of 0 and variance of 1
        # Important after adding the new features above, since they are in a completely
        # different range than the SOAP/MACE descriptors.
        scaler = StandardScaler()
        scaled_feature_matrix = scaler.fit_transform(new_feature_matrix)
    else:
        custom_print('Skipping addition of custom features to feature matrix...')
        scaled_feature_matrix = feature_matrix_X

    # Applying normalization.
    # Normalizing feature matrix before computing the covariance matrix.
    # This will make the inner product behave like a cosine similarity.
    feature_matrix_X = normalize(scaled_feature_matrix, norm='l2', axis=1)

    # Stack the list of feature vectors into a single (n, d) numpy array
    # n = number of samples
    # d = dimension of each feature vector
    n, d = feature_matrix_X.shape

    custom_print('Getting covariance matrix and computing eigenvalues...')

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

    # Ensure eigenvalues are non-negative (numerical stability)
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

    # Compute Shannon entropy of eigenvalues
    # S_shannon = -\sum_{i=1}^n​ \lambda_i​ log(\lambda_i​)
    # Here, we force the convention 0 log(0) = 0 by adding a small epsilon.
    eps = 1e-20
    shannon_entropy = -np.sum(eigenvalues * np.log(eigenvalues + eps))

    # Compute Vendi Score
    # VS=\exp(S_shannon)
    vendi_score = np.exp(shannon_entropy)

    return vendi_score


def tanimoto_distance(struct_1_descriptors, struct_2_descriptors):
    """
    Computes the Tanimoto distance.

    Also known as Jaccard distance.
    """
    tanimoto = 1 - np.sum(struct_1_descriptors * struct_2_descriptors) / (
        np.sum(np.max(struct_1_descriptors, struct_2_descriptors))
    )
    return tanimoto


def cosine_distance(v1, v2):
    """
    Calculates the cosine distance between two real-valued vectors.
    Returns a value between 0 (identical) and 1 (orthogonal for non-negative vectors).
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 1.0

    similarity = dot_product / (norm_v1 * norm_v2)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def get_circles_metric_db(
    dataset: list[Atoms],
    distance_threshold: float,
    descriptor_settings: dict,
    distance_function=cosine_distance,
):
    """Computes the Circles Metric for the database.

    #Circles(S; d, t) := max |C| s.t. d(x, y) > t, ∀x ≠ y ∈ C
    """
    # max |C|
    # What is the maximum number of molecules I can pick from my database
    # such that every molecule I pick is far away form every other molecule
    # I've picked?

    # d(x, y) > t, ∀x ≠ y ∈ C
    # For every possible pair (x,y) of distinct molecules within the set C, the distance
    # between them must be greater than the threshold t.

    # Empty set
    n_structs = len(dataset)
    if n_structs == 0:
        return 0

    # Generate descriptors for the entire dataset
    species = get_species_from_database(dataset)
    desc = SOAP(species=sorted(list(species)), **descriptor_settings)
    descriptors = desc.create(dataset, n_jobs=-1)
    descriptors = normalize(descriptors, norm='l2', axis=1)

    # Pre-compute the full n x n distance matrix for efficiency
    distance_matrix = np.zeros((n_structs, n_structs))
    for i in range(n_structs):
        for j in range(i + 1, n_structs):
            dist = distance_function(descriptors[i], descriptors[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Implement the greedy selection algorithm
    # Randomly reorder the indices of the dataset
    indices = np.arange(n_structs)
    np.random.shuffle(indices)

    # This set C will store the indices of the selected "circle centers"
    selected_indices = []

    for i in indices:
        is_far_enough = True
        # Check the distance to all previously selected molecules
        if selected_indices:
            distances_to_selected = distance_matrix[i, selected_indices]
            # If the minimum distance to an already selected point is not > t, reject it
            if np.min(distances_to_selected) <= distance_threshold:
                is_far_enough = False

        if is_far_enough:
            selected_indices.append(i)

    # The #Circles score is the number of molecules in our final set C
    return len(selected_indices)
