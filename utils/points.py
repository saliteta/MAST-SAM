import torch
import numpy as np
from segment_anything import SamPredictor

def recover_coordinates(matches: np.ndarray, adjusted_H: int, adjusted_W: int, original_H: int, original_W: int) -> torch.Tensor:
    """
    Recover the coordinates of points from an adjusted image size back to the original image size.
    Args:
        matches (torch.Tensor): Tensor of (x, y) coordinates in the adjusted image.
        adjusted_H (int): Height of the adjusted image.
        adjusted_W (int): Width of the adjusted image.
        original_H (int): Height of the original image.
        original_W (int): Width of the original image.
    Returns:
        torch.Tensor: Tensor of (x, y) coordinates in the original image.
    """
    scale_x = int(original_H / adjusted_H)
    scale_y = int(original_W / adjusted_W)
    # convert the current coordinates to the original image size
    original_coords = matches.copy()
    original_coords[:, 0] = (matches[:, 0] * scale_x).astype(int)
    original_coords[:, 1] = (matches[:, 1] * scale_y).astype(int)
    return original_coords



def find_nearest_neighbors(query_points: np.ndarray,
                           reference_points: np.ndarray) -> np.ndarray:
    """
    For each point in query_points, find the nearest neighbor in reference_points 
    via Euclidean distance.

    Args:
        query_points: (Q, 2) array of [x, y].
        reference_points: (R, 2) array of [x, y].

    Returns:
        nn_coords: (Q, 2) array of the nearest-neighbor coordinates in reference_points
                   for each query.
    """
    if len(reference_points) == 0:
        # Edge case: no reference points
        return np.zeros_like(query_points)

    nn_coords = np.zeros_like(query_points)
    
    for i, qp in enumerate(query_points):
        # L2 distance to all reference_points
        dists = np.linalg.norm(reference_points - qp, axis=1)
        min_idx = np.argmin(dists)
        nn_coords[i] = reference_points[min_idx]
    
    return nn_coords


def prompt_sam_with_mask_points(image_cv2: np.ndarray,
                                predictor: SamPredictor,
                                mask_points_list: list,
                                multimask_output=False) -> list:
    """
    For each group of points in `mask_points_list`, call `predictor.predict()` 
    to generate ONE or multiple SAM masks. 
    By default, we use `multimask_output=False` to get a single best mask 
    per group. If you want multiple, set `multimask_output=True`.

    Args:
        image_cv2: (H, W, 3) BGR or RGB image loaded via cv2.
        predictor: A SamPredictor with set_image() already called, or do it here.
        mask_points_list: A list of arrays; each array is shape (k_i, 2) 
                          with coordinates [x, y] for that mask's prompt.
        multimask_output: If False, returns a single mask per group. 
                          If True, returns 3 masks per group by default.

    Returns:
        final_masks: A list of dict or np.ndarray (depending on SAM version),
                     each entry being the mask(s) for that group of points.
                     If `multimask_output=False`, you'll have len(final_masks) == number of groups.
    """
    # If predictor image not set, do it now:
    predictor.set_image(image_cv2) 
    # (But normally you'd do this outside once per image.)

    final_masks = []

    for points in mask_points_list:
        if len(points) == 0:
            # skip empty group
            final_masks.append(None)
            continue

        # SAM expects coords in (y, x) format
        sam_input_points = np.stack([ [pt[1], pt[0]] for pt in points ], axis=0)
        labels = np.ones((len(sam_input_points),), dtype=np.int32)  # all foreground

        masks, scores, logits = predictor.predict(
            point_coords=sam_input_points,
            point_labels=labels,
            multimask_output=multimask_output
        )
        
        # If multimask_output=False, `masks` is shape [H, W]
        # If True, shape [3, H, W]
        # We'll store them in final_masks.  You can also store `scores` if needed.
        final_masks.append(masks)

    return final_masks