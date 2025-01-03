import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from scipy import ndimage

import numpy as np
from cuml.cluster import KMeans as cuKMeans

def distribute_points_among_masks(masks: np.ndarray, total_points: int = 100):
    """
    Distribute `total_points` among the given list of mask arrays 
    proportionally to each mask's area, and cluster points within each 
    mask using GPU-accelerated K-Means.
    
    Args:
        masks (np.ndarray): A boolean mask of shape (k, H, W) where k is the number of masks.
        total_points (int): The total number of prompt points to sample across all masks.
    
    Returns:
        mask_size (np.ndarray): An array containing the size of each mask's prompt points, shape (k).
        all_prompts (np.ndarray): A concatenation of all cluster centers from all masks, shape (N', 2).
    """
    mask_size = []
    all_prompts_list = []
    total_area = masks.sum()
    
    for mask in masks:
        area_m = mask.sum()
        k = int(area_m / total_area * total_points)
        if k == 0:
            continue
        ys, xs = np.where(mask)
        coords = np.column_stack((xs, ys)).astype(np.float32)
        if len(coords) <= k:
            cluster_centers = coords.copy()
        else:
            cu_kmeans = cuKMeans(n_clusters=k, init='random', max_iter=300, random_state=0)
            cu_kmeans.fit(coords)
            cluster_centers = cu_kmeans.cluster_centers_
        cluster_centers = cluster_centers.astype(np.int32)
        mask_size.append(len(cluster_centers))
        all_prompts_list.append(cluster_centers)
    
    if all_prompts_list:
        all_prompts = np.concatenate(all_prompts_list, axis=0)
    else:
        all_prompts = np.zeros((0, 2), dtype=np.int32)
    
    return mask_size, all_prompts

def visualize_mask_points(first_image_cv2, mask_points_list, cmap_name='rainbow'):
    """
    Plot all points from all masks at once (single ax.scatter() call).
    
    Each mask's points share a distinct color, but we don't call scatter repeatedly.
    
    Args:
        first_image_cv2 (np.ndarray): The original image in (H, W, 3) format.
        mask_points_list (List[np.ndarray]): A list of arrays of shape (k_i, 2).
            Each array stores the [x, y] points for one mask.
        cmap_name (str): A valid matplotlib colormap name (e.g. 'rainbow', 'viridis', etc.).
    """
    # Create a new figure
    fig, ax = plt.subplots()
    ax.imshow(first_image_cv2)

    # Prepare a colormap
    cmap = plt.get_cmap(cmap_name)
    n_masks = len(mask_points_list)

    # We will flatten all points from all masks into a single array,
    # along with a matching color array of RGBA values.
    all_points = []
    all_colors = []

    for i, points in enumerate(mask_points_list):
        # Get a distinct color from the colormap
        # e.g. i/(n_masks-1) to spread them evenly
        color = cmap(i / max(1, (n_masks - 1)))
        # color is RGBA, something like (r, g, b, a)

        for (x, y) in points:
            all_points.append([x, y])
            all_colors.append(color)

    # Convert to arrays
    all_points = np.array(all_points)    # shape (N, 2)
    all_colors = np.array(all_colors)    # shape (N, 4)

    # Single scatter call
    ax.scatter(
        all_points[:, 0],
        all_points[:, 1],
        c=all_colors,
        s=15,
        marker='o'
    )

    ax.set_title("All Mask Points in One Scatter Call")
    plt.show()
    plt.savefig('mask_points')
    plt.close(fig)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

def merge_masks(masks: np.ndarray, threshold=0.5) -> np.ndarray:
    """
    Merge masks in `masks` if they overlap more than `threshold`
    fraction of the smaller mask's area.
    
    Args:
        masks (np.ndarray): A list of mask arrays.
        threshold (float): Overlap fraction w.r.t. the smaller area required to merge.
        
    Returns:
        (np.ndarray): A new array of merged masks.
    """
    n = len(masks)
    uf = UnionFind(n)
    
    # Determine which masks should be merged
    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(masks[i], masks[j], threshold):
                uf.union(i, j)
    
    # Group masks by their root parent
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups[root].append(i)
    
    # Merge masks within each group
    merged_masks = []
    for group in groups.values():
        # Merge all masks in the group into one
        merged_seg = np.zeros_like(masks[group[0]], dtype=bool)
        for idx in group:
            merged_seg |= masks[idx]
        merged_masks.append(merged_seg)
    
    return np.array(merged_masks)

def should_merge(segA: np.ndarray, segB: np.ndarray, threshold=0.5):
    """
    Decide if two masks should merge based on how large their overlap is
    relative to the smaller mask's area.
    """
    overlap = segA & segB
    overlap_area = np.sum(overlap)
    smaller_area = min(np.sum(segA), np.sum(segB))
    
    if smaller_area == 0:
        return False
    
    overlap_ratio = overlap_area / smaller_area
    return overlap_ratio > threshold


def refine_masks(masks: List[np.ndarray], update = False) -> np.ndarray:
    """
    Refine the masks by first merging the overlapping masks, and then creating new masks
    by finding the unmasked region. Any continuous unmasked region (no existing mask
    separating it) is turned into a new mask if it has area >= 100 pixels.

    Args:
        masks: (n, H, W) masks generated by SAM (boolean or 0/1 float/uint)

    Returns:
        masks: (m, H, W) the refined set of masks, including newly added ones that cover
               large continuous unmasked areas.
    """
    # 1) Merge overlapping masks (depends on your implementation of merge_masks).
    #    After this step, 'masks' should still be shape (n, H, W).
    masks = merge_masks(masks, threshold=0.5)
    
    
    # Make sure masks are in [0,1] or boolean form
    masks_bool = (masks > 0).astype(np.uint8)  # shape: (n, H, W)
    
    masks_bool = filter_masks(masks_bool, connectivity=8)
    

    if update == False:
        return masks_bool
    
    # 2) Find the union of these merged masks across the channel dimension
    #    merged_union will be shape (H, W), indicating which pixels are covered by any mask.
    merged_union = np.any(masks_bool, axis=0).astype(np.uint8)
    

    # 3) Identify the unmasked region (pixels not covered by any mask)
    unmasked_region:np.ndarray = (1 - merged_union).astype(np.uint8)
    
    # 4) Find connected components in the unmasked region
    num_labels, labels = cv2.connectedComponents(unmasked_region, connectivity=8)
    #   labels will have values from 0 .. num_labels-1, where 0 = background

    new_masks = []
    for label_idx in range(1, num_labels):  # skip background = 0
        region_mask = (labels == label_idx).astype(np.uint8)  # shape: (H, W)
        # Only keep regions with size >= 100 pixels
        if region_mask.sum() >= 1000:
            new_masks.append(region_mask)

    # 5) Combine any newly found masks with the existing masks
    if len(new_masks) == 0:
        # No unmasked region large enough to become a new mask
        return masks
    else:
        # Stack new masks along axis=0 to match shape (m, H, W)
        new_masks = np.stack(new_masks, axis=0)  # shape: (k, H, W)
        # Concatenate them with the existing (merged) masks
        # Note that 'masks' might need to be cast to uint8 if it's float/bool
        refined_masks = np.concatenate([masks_bool, new_masks], axis=0)
        return refined_masks


def filter_masks(masks, connectivity=8):
    """
    Filters the input masks to retain only the largest continuous region.
    
    Parameters:
        masks (numpy.ndarray): A 3D numpy array of shape (n, h, w) with dtype=bool.
        connectivity (int): Specifies connectivity, either 4 or 8.
        
    Returns:
        numpy.ndarray: The filtered masks with the same shape and dtype as input.
    """
    if connectivity == 8:
        structure = np.ones((3,3), dtype=int)
    elif connectivity == 4:
        structure = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=int)
    else:
        raise ValueError("Connectivity must be 4 or 8.")
    
    for i in range(len(masks)):
        mask = masks[i]
        labeled_array, num_features = ndimage.label(mask, structure=structure)
        if num_features == 0:
            continue  # mask is all False, no action needed
        # Calculate sizes of each component
        component_sizes = ndimage.sum(mask, labeled_array, index=np.arange(1, num_features+1))
        # Find the label of the largest component
        largest_component_label = np.argmax(component_sizes) + 1  # labels start at 1
        # Create a mask with only the largest component
        filtered_mask = (labeled_array == largest_component_label)
        # Update the original mask
        masks[i] = filtered_mask
    return masks