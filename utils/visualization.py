import matplotlib.pyplot as plt
from random import randint
import numpy as np
import torch

from typing import Dict, List
from pathlib import Path

def visualize_correspondences(
    image_0: torch.Tensor,
    image_1: torch.Tensor,
    coords_0: np.ndarray,
    coords_1: np.ndarray,
    points_correlation: int = 20,
    save_path: str = 'correspondences_visualization.png'
):
    """
    Visualize the correspondences between two images by drawing matching lines.
    
    Args:
        image_0 (torch.Tensor): The first image tensor of shape (1, 3, H, W).
        image_1 (torch.Tensor): The second image tensor of shape (1, 3, H, W).
        coords_0 (np.ndarray): Coordinates in the first image, shape (N, 2).
        coords_1 (np.ndarray): Coordinates in the second image, shape (N, 2).
        points_correlation (int): How many random correspondences to visualize.
        save_path (str): Where to save the final visualization.
    """
    # Convert the tensors to NumPy (H, W, C)
    img0 = (image_0.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1 )/2
    img1 = (image_1.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1 )/2
    
    # Pick 'points_correlation' random indices
    assert len(coords_0) == len(coords_1), "coords_0 and coords_1 must have the same length."
    indices = [randint(0, len(coords_0) - 1) for _ in range(points_correlation)]
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
    plt.show()
    
    
def show_anns(image_cv2: np.ndarray, anns: List[Dict], figure_location: Path, alpha: float = 0.35):
    
    
    fig, ax = plt.subplots()
    # Show image (Matplotlib expects RGB ordering)
    ax.imshow(image_cv2)

    
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    print(len(sorted_anns))
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[m] = color_mask
    ax.imshow(img)
    plt.savefig(figure_location)