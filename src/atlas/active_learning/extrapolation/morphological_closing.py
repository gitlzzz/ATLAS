"""Utilities for boundary determination using morphological closing."""

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from skimage import measure
from skimage.morphology import closing, disk

from atlas.core.code_utils import custom_print


def create_image_mask(
    data_X, data_Y, disk_size=10, figsize=(10, 8), dpi=100, threshold=250, point_size=5
):
    """Creates mask for 2D points by applying morphological closing to plot."""
    fig_seg, ax_seg = plt.subplots(figsize=figsize)
    fig_seg.patch.set_facecolor('white')
    ax_seg.set_facecolor('white')

    # Plot the raw points in black for maximum contrast
    ax_seg.scatter(data_X, data_Y, s=point_size, c='black')
    ax_seg.axis('off')

    # Bridge Figure to PIL Image buffer
    buf = io.BytesIO()
    fig_seg.savefig(buf, format='png', dpi=dpi, facecolor='white')
    buf.seek(0)

    # Load image as a grayscale numpy array
    img_gray = np.array(Image.open(buf).convert('L'))
    img_h, img_w = img_gray.shape

    # Thresholding: Anything darker than light gray is a data point
    dots_mask = img_gray < threshold

    # Morphological Closing: Bridge the gaps between the dots to form a solid blob
    solid_mask = closing(dots_mask, disk(disk_size))

    return solid_mask, img_w, img_h, fig_seg, ax_seg


def filter_points_by_mask(data_X, data_Y, solid_mask, ax_seg, img_w, img_h):
    """Filters points based on whether they fall inside the generated mask."""
    selected_indices = []

    for i in range(len(data_X)):
        x_val, y_val = data_X[i], data_Y[i]

        # Transforms: Data -> Screen Pixels -> Figure Fractional -> Mask Pixels
        screen_coords = ax_seg.transData.transform((x_val, y_val))
        fig_coords = ax_seg.figure.transFigure.inverted().transform(screen_coords)

        pixel_x = int(np.floor(fig_coords[0] * img_w))
        pixel_y = int(np.floor((1 - fig_coords[1]) * img_h))  # Invert Y

        # Check if the mapped pixel falls within our generated mask
        if (
            0 <= pixel_x < img_w
            and 0 <= pixel_y < img_h
            and solid_mask[pixel_y, pixel_x]
        ):
            selected_indices.append(i)

    filtered_X = data_X[selected_indices]
    filtered_Y = data_Y[selected_indices]

    return selected_indices, filtered_X, filtered_Y


def extract_boundaries_from_mask(
    solid_mask, fig_seg, ax_seg, img_w, img_h, contour_level=0.5
):
    """Extracts data-space boundaries from the pixel mask."""
    # Pad the mask with 1 pixel of 'False' on all sides to ensure the contour
    # algorithm finds a closed loop even if the shape touches the image border.
    padded_mask = np.pad(
        solid_mask, pad_width=1, mode='constant', constant_values=False
    )

    # Find contours on the padded mask
    pixel_contours = measure.find_contours(padded_mask, level=contour_level)
    data_boundaries = []
    inv_transData = ax_seg.transData.inverted()

    for contour in pixel_contours:
        # Subtract 1 to remove the artificial padding offset we just added
        y_pixels = contour[:, 0] - 1
        x_pixels = contour[:, 1] - 1

        # Reverse Mask Pixels to Figure-Fractional (0.0 to 1.0)
        fig_x = x_pixels / img_w
        fig_y = 1.0 - (y_pixels / img_h)  # Invert Y back to standard plot coordinates

        fig_coords = np.column_stack([fig_x, fig_y])

        # Reverse Figure-Fractional to Screen Pixels
        screen_coords = fig_seg.transFigure.transform(fig_coords)

        # Reverse Screen Pixels to Data Coordinates
        data_coords = inv_transData.transform(screen_coords)
        data_boundaries.append(data_coords)

    return data_boundaries


def process_morphological_closing(
    data_X, data_Y, disk_size=1, figsize=(10, 8), dpi=100, threshold=250, point_size=5
) -> dict:
    """
    Extract shape and filter points using morphological closing.

    Returns a dictionary containing the filtered points, selected indices,
    boundaries, and the generated mask.
    """
    solid_mask, img_w, img_h, fig_seg, ax_seg = create_image_mask(
        data_X,
        data_Y,
        disk_size=disk_size,
        figsize=figsize,
        dpi=dpi,
        threshold=threshold,
        point_size=point_size,
    )

    selected_indices, filtered_X, filtered_Y = filter_points_by_mask(
        data_X, data_Y, solid_mask, ax_seg, img_w, img_h
    )

    data_boundaries = extract_boundaries_from_mask(
        solid_mask, fig_seg, ax_seg, img_w, img_h
    )

    # Clean up memory
    plt.close(fig_seg)

    return {
        'selected_indices': selected_indices,
        'filtered_X': filtered_X,
        'filtered_Y': filtered_Y,
        'data_boundaries': data_boundaries,
        'solid_mask': solid_mask,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Extract shapes and boundaries from 2D points using morphological closing.'
        )
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input numpy array file (e.g., latent_space_array.npy)',
    )
    parser.add_argument(
        '--disk-size',
        type=int,
        default=10,
        help='Disk size for morphological closing. Increase to capture wider outliers.',
    )
    parser.add_argument(
        '--save_hulls',
        action='store_true',
        help='Save the extracted boundaries as a pickled list of arrays.',
    )
    args = parser.parse_args()

    import os

    base_name = os.path.splitext(os.path.basename(args.input_file))[0]

    custom_print('Loading latent space...', 'info')
    latent_space_array = np.load(args.input_file, allow_pickle=True)
    data_X = latent_space_array[:, 0]
    data_Y = latent_space_array[:, 1]

    custom_print('Processing morphological closing...', 'info')
    results = process_morphological_closing(data_X, data_Y, disk_size=args.disk_size)

    filtered_X = results['filtered_X']
    filtered_Y = results['filtered_Y']
    data_boundaries = results['data_boundaries']
    solid_mask = results['solid_mask']

    num_filtered = len(filtered_X)
    custom_print(
        f'Data conversion complete! Selected {num_filtered} of {len(data_X)} '
        f'({(num_filtered / len(data_X)) * 100:.1f} %) points.',
        'done',
    )
    custom_print(
        f'Extracted {len(data_boundaries)} exact image boundary polygons.', 'done'
    )

    if args.save_hulls:
        custom_print('Exporting boundaries to NumPy array...', 'info')
        np.save(
            f'all_cluster_boundaries_{base_name}.npy',
            np.array(data_boundaries, dtype=object),
        )

    plt.figure(figsize=(10, 8))
    plt.scatter(
        data_X,
        data_Y,
        s=5,
        c='lightgray',
        label='All Points',
    )
    total_area = 0.0
    for boundary in data_boundaries:
        boundary = np.atleast_2d(boundary)
        if len(boundary) < 3:
            continue

        poly = Polygon(boundary)
        total_area += poly.area

        plt.plot(
            boundary[:, 0],
            boundary[:, 1],
            color='red',
            linewidth=2,
            label='Extracted Boundary',
        )

    custom_print(f'Total extracted area: {total_area:.2f}', 'info')

    plt.text(
        0.05,
        0.95,
        f'Total Area: {total_area:.2f}',
        transform=plt.gca().transAxes,
        color='black',
        fontsize=14,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=5),
    )
    plt.title(base_name, fontsize=16)

    save_path_boundary = f'boundary_overlay_{base_name}.png'
    plt.savefig(save_path_boundary, bbox_inches='tight', dpi=150)

    custom_print(f"Boundary overlay plot saved as '{save_path_boundary}'", 'done')
