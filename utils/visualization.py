import matplotlib.pyplot as plt
from random import randint
import numpy as np
import torch

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
    
    
def show_anns(image_cv2: np.ndarray, anns: List[np.ndarray], figure_location: Path, alpha: float = 0.35):
    
    
    fig, ax = plt.subplots()
    # Show image (Matplotlib expects RGB ordering)
    ax.imshow(image_cv2)

    
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns[0].shape[0], anns[0].shape[1], 4))
    img[:, :, 3] = 0
    for ann in anns:
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[ann] = color_mask
    ax.imshow(img)
    plt.savefig(figure_location)
    plt.close()