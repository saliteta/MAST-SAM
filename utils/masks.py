import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def distribute_points_among_masks(masks, total_points=100):
    """
    Distribute `total_points` among the given list of mask dictionaries 
    proportionally to each mask's area, and cluster points within each 
    mask using K-Means.
    
    Args:
        masks (List[dict]): A list of mask dictionaries.
            Each dictionary has keys like:
               - 'segmentation': (H, W) boolean array
               - 'area': float
               - ... possibly other keys
        total_points (int): The total number of prompt points we want to sample
                            across *all* masks.
                            
    Returns:
        mask_size (np.ndarray): 
            An arrays containing the size of each mask, shape (k).
            
        all_prompts (np.ndarray): 
            A concatenation of all cluster centers from all masks, shape (N', 2).
            Where N' ≈ total_points.
    """
    # 1) Compute total area of all masks
    total_area = sum(m['area'] for m in masks)
    if total_area <= 0:
        return [], np.zeros((0, 2), dtype=np.int32)
    
    mask_size = []
    all_prompts_list = []
    
    for m in masks:
        area_m = m['area']
        seg = m['segmentation']  # boolean numpy array of shape (H, W)
        
        # 2) Determine how many prompts to allocate to this mask
        #    proportional to area
        frac = area_m / total_area
        k_m = int(round(frac * total_points))
        
        # Optionally enforce min. or max. prompts per mask
        # e.g. at least 1 if there's enough area
        # or skip if k_m == 0:
        if k_m < 1:
            continue
        
        # 3) Collect all (x, y) points in the mask
        #    m['segmentation'] = True where the pixel belongs to the mask
        ys, xs = np.where(seg == True)
        if len(xs) == 0 or len(ys) == 0:
            # no pixels in mask
            continue
        
        # Convert to float, shape (num_pixels, 2)
        coords = np.column_stack((xs, ys)).astype(np.float32)
        
        # 4) If the # of mask pixels < k_m, we trivially use all points
        #    or simply reduce k_m to len(coords)
        if len(coords) <= k_m:
            cluster_centers = coords
        else:
            # K-Means clustering to find k_m representative centers
            kmeans = KMeans(n_clusters=k_m, random_state=0)
            kmeans.fit(coords)
            cluster_centers = kmeans.cluster_centers_
        
        # Round/convert to integer pixel coords
        cluster_centers = cluster_centers.astype(np.int32)
        
        mask_size.append(len(cluster_centers))
        all_prompts_list.append(cluster_centers)
    
    # 5) Concatenate all prompt points across all masks
    if len(all_prompts_list) > 0:
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

def merge_masks(masks, threshold=0.5):
    """
    Merge masks in `masks` if they overlap more than `threshold` 
    fraction of the smaller mask's area.
    
    Args:
        masks (List[dict]): List of dicts, each containing:
            - 'segmentation': (H, W) bool array
            - 'area': float
            - 'bbox': List or tuple [x_min, y_min, x_max, y_max]
            - 'predicted_iou', 'point_coords', 'stability_score', 'crop_box' (optional)
        threshold (float): Overlap fraction w.r.t. the smaller area required to merge.
        
    Returns:
        (List[dict]): A new list of merged mask dictionaries.
    """
    # We iteratively try to merge until no merges happen in a full pass
    merged = True
    masks_out = masks[:]  # make a copy

    while merged:
        merged = False
        new_masks = []
        i = 0
        while i < len(masks_out):
            # Compare mask i with all subsequent masks to check potential merge
            j = i + 1
            has_merged = False
            while j < len(masks_out):
                if should_merge(masks_out[i], masks_out[j], threshold):
                    # Merge these two
                    merged_mask = merge_two_masks(masks_out[i], masks_out[j])
                    new_masks.append(merged_mask)
                    
                    # Remove them from the list
                    del masks_out[j]
                    del masks_out[i]
                    
                    merged = True
                    has_merged = True
                    break  # break the inner loop
                else:
                    j += 1
            
            if not has_merged:
                # If mask i did not merge with anything, keep it
                new_masks.append(masks_out[i])
                i += 1
        
        masks_out = new_masks
    
    return masks_out

def should_merge(maskA, maskB, threshold=0.5):
    """
    Decide if two masks should merge based on how large their overlap is 
    relative to the smaller mask's area.
    """
    segA = maskA['segmentation']
    segB = maskB['segmentation']
    
    overlap = segA & segB
    overlap_area = overlap.sum()
    smaller_area = min(maskA['area'], maskB['area'])
    
    if smaller_area == 0:
        return False
    
    # Overlap ratio = overlap_area / area_of_smaller_mask
    overlap_ratio = overlap_area / smaller_area
    return (overlap_ratio > threshold)

def merge_two_masks(maskA, maskB):
    """
    Merge two masks' dicts into a single dict 
    by taking the union of their 'segmentation'.
    """
    segA = maskA['segmentation']
    segB = maskB['segmentation']
    
    # Union of A and B
    new_seg = segA | segB
    new_area = new_seg.sum()

    # We can combine bounding boxes by taking min/max
    bboxA = maskA.get('bbox', None)
    bboxB = maskB.get('bbox', None)
    new_bbox = _merge_bbox(bboxA, bboxB, new_seg.shape)  # see helper below

    # predicted_iou, stability_score, etc.: pick a simple approach
    predicted_iou = max(maskA.get('predicted_iou', 0), maskB.get('predicted_iou', 0))
    stability_score = max(maskA.get('stability_score', 0), maskB.get('stability_score', 0))

    # You could union 'point_coords' or discard it—depending on usage
    point_coordsA = maskA.get('point_coords', [])
    point_coordsB = maskB.get('point_coords', [])
    new_point_coords = list(point_coordsA) + list(point_coordsB)

    # Similarly, you can pick how to handle 'crop_box'
    # For example, unify them or just pick the bigger region
    new_crop_box = None
    if 'crop_box' in maskA or 'crop_box' in maskB:
        # Example: just unify them or pick whichever
        new_crop_box = _merge_bbox(maskA.get('crop_box', None),
                                   maskB.get('crop_box', None),
                                   new_seg.shape)

    merged_mask = {
        'segmentation': new_seg,
        'area': float(new_area),
        'bbox': new_bbox,
        'predicted_iou': predicted_iou,
        'stability_score': stability_score,
        'point_coords': new_point_coords,
        'crop_box': new_crop_box
    }
    return merged_mask

def _merge_bbox(bboxA, bboxB, shape):
    """
    Merge bounding boxes by taking min of x_min, y_min 
    and max of x_max, y_max across both. 
    If None, we compute from the segmentation as fallback.
    """
    # If both bboxes are None, we might compute from shape or from the segmentation
    if bboxA is None and bboxB is None:
        return None
    if bboxA is None:
        return bboxB
    if bboxB is None:
        return bboxA

    # BBox format assumed: [x_min, y_min, x_max, y_max]
    x_min = min(bboxA[0], bboxB[0])
    y_min = min(bboxA[1], bboxB[1])
    x_max = max(bboxA[2], bboxB[2])
    y_max = max(bboxA[3], bboxB[3])

    # Clip to the image boundary if needed
    H, W = shape
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, W - 1)
    y_max = min(y_max, H - 1)

    return [x_min, y_min, x_max, y_max]