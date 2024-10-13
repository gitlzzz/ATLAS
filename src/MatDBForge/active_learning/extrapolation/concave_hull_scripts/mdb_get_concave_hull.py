#!/usr/bin/env python3
"""Utilities to compute the concave hull for a point cloud of atomic descriptors."""

import matplotlib.pyplot as plt
import numpy as np
from juliacall import Main as jl
from juliacall import convert as jl_convert
from shapely.geometry import Point, Polygon


def get_concave_hull_julia(latent_space: np.ndarray) -> np.ndarray:
    # Load the required Julia modules.
    jl.seval("using GMT")

    if len(latent_space.shape) > 2:
        latent_space = np.vstack(latent_space)

    # Convert the latent space to a Julia array.
    # As it is necessary for the concave hull function.
    latent_space = jl_convert(jl.Matrix[jl.Float32], latent_space)

    # Compute the concave hull using the GMT.jl package.
    concave_hull = jl.concavehull(latent_space, 0.075)

    # Convert the result back to a NumPy array.
    concave_hull = np.array(concave_hull)

    return concave_hull


def check_atom_in_domain(
    concave_hull: np.ndarray, descriptors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    polygon = Polygon(concave_hull)
    for point in descriptors:
        p = Point(point)
        if polygon.contains(p):
            point_inside.append(point)
            all_points_in_out.append(True)
        else:
            point_outside.append(point)
            all_points_in_out.append(False)

    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)
    all_points_in_out = np.array(all_points_in_out)
    return point_inside, point_outside, all_points_in_out


def plot_concave_hull(
    concave_hull: np.ndarray,
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
    plt.title("Concave Hull")
    plt.xlabel("x")
    plt.legend()
    plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    plot_hull = True
    latent_space_file = "latent_space.npy"

    # Gather the latent space from the autoencoder.
    print("Reading latent space...")
    latent_space = np.load(latent_space_file)
    print("Latent space read.")

    # Compute the concave hull using Julia.
    print("Computing concave hull...")
    concave_hull = get_concave_hull_julia(latent_space)
    np.save("concave_hull.npy", concave_hull)
    print("Concave hull computed, saved to 'concave_hull.npy'.")

    print("Plotting concave hull...")
    if plot_hull:
        plot_concave_hull(
            concave_hull=concave_hull,
            latent_space=latent_space,
            filename="concave_hull.png",
        )
    print("Concave hull plotted.")
    print("Calculation done.")
