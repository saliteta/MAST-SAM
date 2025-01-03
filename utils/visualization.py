import matplotlib.pyplot as plt
from random import randint
import numpy as np
import torch
from matplotlib.colors import Normalize

from typing import Dict, List
from pathlib import Path

def visualize_correspondences(
    img0: torch.Tensor,
    img1: torch.Tensor,
    coords_0: np.ndarray,
    coords_1: np.ndarray,
    points_correlation: int = 20,
    save_path: str = 'correspondences_visualization.png'
):
    """
    Visualize the correspondences between two images by drawing matching lines.
    
    Args:
        image_0 (torch.Tensor): The first image tensor of shape (H, W, 3).
        image_1 (torch.Tensor): The second image tensor of shape (H, W, 3).
        coords_0 (np.ndarray): Coordinates in the first image, shape (N, 2).
        coords_1 (np.ndarray): Coordinates in the second image, shape (N, 2).
        points_correlation (int): How many random correspondences to visualize.
        save_path (str): Where to save the final visualization.
    """
    
    # Pick 'points_correlation' random indices
    assert len(coords_0) == len(coords_1), "coords_0 and coords_1 must have the same length."
    if points_correlation > 0: 
        indices = [randint(0, len(coords_0) - 1) for _ in range(points_correlation)]
    else:
        indices = range(len(coords_0))
    coords_0_sub = coords_0[indices]
    coords_1_sub = coords_1[indices]
    
    # Prepare a canvas by concatenating images side-by-side
    h0, w0, c0 = img0.shape
    h1, w1, c1 = img1.shape
    height = max(h0, h1)
    width = w0 + w1
    
    # Create the blank canvas
    canvas = np.zeros((height, width, 3), dtype=img0.dtype)
    
    # Place the two images on the canvas
    canvas[:h0, :w0, :] = img0
    canvas[:h1, w0:w0 + w1, :] = img1
    
    # Plot everything in one figure
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(canvas)
    ax.set_axis_off()
    
    # First image points: (x0, y0)
    # Second image points: (x1 + w0, y1) – we must shift by w0 horizontally
    for (x0, y0), (x1, y1) in zip(coords_0_sub, coords_1_sub):
        # Draw the keypoints
        ax.plot(x0, y0, 'ro', markersize=4)  # Red dot on image 0
        ax.plot(x1 + w0, y1, 'bo', markersize=4)  # Blue dot on image 1
        
        # Draw a line connecting them
        ax.plot([x0, x1 + w0], [y0, y1], color='yellow', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    

def show_anns(
    image_cv2: np.ndarray,
    pred_anns: List[np.ndarray],
    points_mask: List[np.ndarray],
    gt_ann: np.ndarray,
    figure_location: Path,
    alpha: float = 0.35
):
    """
    Display two images side by side:
      - Left: original image with predicted annotations overlaid (colormap with alpha).
      - Right: original image with ground truth annotation (colormap with alpha).
    Also display scatter points on the images with colors matching the colormap.

    :param image_cv2:        The original image (NumPy array in BGR or RGB).
    :param pred_anns:        List of predicted annotations; each item is a Boolean or index mask.
    :param points_mask:      List of point sets, each set is an array of points [[x1,y1],[x2,y2],...].
    :param gt_ann:           Single ground-truth annotation mask (Boolean or index mask).
    :param figure_location:  Path where the resulting figure is saved.
    :param alpha:            Transparency level (0=fully transparent, 1=fully opaque).
    """
    # Convert BGR to RGB if necessary
    if image_cv2.shape[2] == 3:
        import cv2
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_cv2

    # Prepare left and right images
    left_img = image_rgb.copy()
    right_img = image_rgb.copy()

    # Choose a colormap
    cmap = plt.get_cmap('jet')
    norm = Normalize(vmin=0, vmax=len(pred_anns))

    # Create a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Build RGBA overlay for predicted annotations
    overlay_pred = np.zeros((left_img.shape[0], left_img.shape[1], 4), dtype=np.float32)
    overlay_pred[..., 3] = 0  # Initialize alpha to 0

    for idx, ann in enumerate(pred_anns):
        color = cmap(norm(idx))
        overlay_pred[ann] = [color[0], color[1], color[2], alpha]

    # Build RGBA overlay for ground truth annotation
    overlay_gt = np.zeros((right_img.shape[0], right_img.shape[1], 4), dtype=np.float32)
    overlay_gt[..., 3] = 0

    if gt_ann is not None:
        # Assign a specific color to ground truth, e.g., red
        color_gt = cmap(norm(len(pred_anns)))  # or any other logic
        overlay_gt[gt_ann == 1] = [1.0, 0.0, 0.0, alpha]

    # Plot the images and overlays
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 8))

    # Left: original + predicted overlay
    ax_left.imshow(left_img)
    ax_left.imshow(overlay_pred)
    ax_left.set_title('Predicted Annotations')
    ax_left.axis('off')

    # Plot points on the left image
    for i, points in enumerate(points_mask):
        if points is not None and len(points) > 0:
            points = np.array(points)
            color = cmap(norm(i))
            ax_left.scatter(points[:, 0], points[:, 1], color=color, s=10)

    # Right: original + ground-truth overlay
    ax_right.imshow(right_img)
    ax_right.imshow(overlay_gt)
    ax_right.set_title('Ground Truth')
    ax_right.axis('off')

    # Add color bar
    cbar = fig.colorbar(sm, ax=[ax_left, ax_right], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Mask Index')

    plt.tight_layout()
    plt.savefig(figure_location)
    plt.close()  
    
def show_gt_as_binary(gt_ann: np.ndarray, figure_location: Path):
    
    
    
    """
    Display the ground-truth mask as a binary image (black and white),
    and save the resulting figure to disk.

    :param gt_ann:         A 2D numpy array representing the ground-truth mask
                           (boolean or 0/1 values).
    :param figure_location: Path where the figure is saved.
    """
    # Create a new figure
    plt.figure(figsize=(6, 6))
    
    # Display the mask in black and white
    #   - `cmap='gray'` uses grayscale
    #   - `vmin=0` and `vmax=1` ensure it’s treated as a 0–1 binary mask
    plt.imshow(gt_ann, cmap='gray', vmin=0, vmax=1)
    
    # Remove axis ticks/labels
    plt.axis('off')
    plt.title("Ground Truth Mask (Binary)")
    
    # Save and close
    plt.savefig(figure_location, bbox_inches='tight', pad_inches=0)
    plt.close()
    

def visualize_conf(hm1: np.ndarray, hm2: np.ndarray, save_path: str = 'heatmaps.png'):
    """
    Display two heat maps side by side.
    Each heat map has shape (1, H, W), so we squeeze to (H, W).

    :param hm1: First heat map, shape (1, H, W).
    :param hm2: Second heat map, shape (1, H, W).
    """
    # 1) Squeeze out the extra dimension -> (H, W)
    hm1_squeezed = hm1.squeeze()  # from (1, H, W) to (H, W)
    hm2_squeezed = hm2.squeeze()  # from (1, H, W) to (H, W)

    # 2) Create a figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    # 3) Display each heat map
    ax1.imshow(hm1_squeezed, cmap='jet', aspect='auto')
    ax1.set_title("Heat Map 1")
    ax1.axis('off')

    ax2.imshow(hm2_squeezed, cmap='jet', aspect='auto')
    ax2.set_title("Heat Map 2")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    plt.close()
    
def visualize_masks(masks: List[np.ndarray], cmap: str = 'tab20', save_place = 'masks.png'):
    # Check that all masks have the same shape
    first_shape = masks[0].shape
    if not all(mask.shape == first_shape for mask in masks):
        raise ValueError("All masks must have the same shape.")
    # Create a label image
    label_img = np.zeros(first_shape, dtype=np.int32)
    for i, mask in enumerate(masks):
        label_img[mask] = i + 1  # labels start from 1
    # Create a normalized colormap
    norm = plt.Normalize(1, len(masks))
    # Display the label image with the colormap
    plt.imshow(label_img, cmap=cmap, norm=norm)
    plt.colorbar(ticks=np.arange(1, len(masks)+1))
    plt.axis('off')
    plt.show()
    plt.savefig(save_place)
    plt.close()