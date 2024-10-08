"""Utilities to compute the concave hull for a point cloud of atomic descriptors."""

import matplotlib.pyplot as plt
import numpy as np
from juliacall import Main as jl
from juliacall import convert as jl_convert
from shapely.geometry import Point, Polygon


def get_concave_hull_julia(latent_space: np.ndarray) -> np.ndarray:
    # Load the required Julia modules.
    jl.seval("using GMT")

    # Convert the latent space to a Julia array.
    # As it is necessary for the concave hull function.
    latent_space = jl_convert(jl.Matrix[jl.Float32], latent_space)

    # Compute the concave hull using the GMT.jl package.
    concave_hull = jl.concavehull(latent_space, 0.075)

    # Convert the result back to a NumPy array.
    concave_hull = np.array(concave_hull)

    return concave_hull


def check_atom_in_domain(
    concave_hull: np.ndarray, random_points: np.ndarray
) -> np.ndarray:
    point_inside = []
    point_outside = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    polygon = Polygon(concave_hull)
    for point in random_points:
        p = Point(point)
        if polygon.contains(p):
            point_inside.append(point)
        else:
            point_outside.append(point)

    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)
    return point_inside, point_outside


def plot_concave_hull(
    concave_hull: np.ndarray,
    point_inside: np.ndarray,
    point_outside: np.ndarray,
    latent_space: np.ndarray,
    filename: str = "concave_hull.png",
):
    # Plotting the concave hull in 2D space using lines
    plt.plot(concave_hull[:, 0], concave_hull[:, 1], "r-")
    plt.plot(
        latent_space[:, 0],
        latent_space[:, 1],
        "o",
        markersize=2,
        alpha=0.5,
        label="Descriptor in database",
        markeredgewidth=0,
        color="#b16286",
    )
    plt.plot(
        point_inside[:, 0],
        point_inside[:, 1],
        "s",
        label="Structure in domain",
        color="#8ec07c",
        markersize=5,
        markeredgewidth=1.5,
        markeredgecolor="#282828",
    )
    plt.plot(
        point_outside[:, 0],
        point_outside[:, 1],
        "s",
        label="Structure out of domain",
        color="#fb4934",
        markersize=5,
        markeredgewidth=1.5,
        markeredgecolor="#282828",
    )
    plt.title("Concave Hull")
    plt.xlabel("x")
    plt.legend()
    plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    plot_hull = True

    # Set the random seed for reproducibility.
    rng = np.random.default_rng(seed=420)

    # Gather the latent space from the autoencoder.
    latent_space = np.load("./convex_hull/autoencoder/latent_space.npy")

    # Generating 50 random points
    random_points = np.random.ranf(size=(50, 2))
    random_points = (2 - -4) * random_points + -4

    # Compute the concave hull using Julia.
    concave_hull = get_concave_hull_julia(latent_space)

    # Check if the random points are inside the concave hull.
    point_inside, point_outside = check_atom_in_domain(concave_hull, random_points)

    if plot_hull:
        plot_concave_hull(
            concave_hull,
            point_inside,
            point_outside,
            latent_space,
            "/tmp/concave_hull.png",
        )
