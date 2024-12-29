import sys
import os
import numpy as np  


import mast3r.utils.path_to_dust3r
import argparse
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

from dust3r.inference import inference
from dust3r.utils.image import load_images
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from utils.masks import distribute_points_among_masks, merge_masks
from utils.visualization import visualize_correspondences, show_anns
from utils.points import recover_coordinates, prompt_sam_with_mask_points, find_corresponding_neighbors

from evaluation import calculateMetrics
import pandas as pd

from typing import List
# from evaluation import Cal_IOU

def parser():
    parser = argparse.ArgumentParser("SAM Encoder Test", add_help=True)
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, default="/home/xiongbutian/workspace/Foundation_Models/SAM/sam_vit_h_4b8939.pth", required=False, help="path to sam checkpoint file")
    parser.add_argument("--mast3r_checkpoint", type=str, default='/home/xiongbutian/workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', required=False, help="path to sam checkpoint file")
    parser.add_argument("--image_dir",nargs='+', required=True, help="paths to image folders")
    parser.add_argument("--gt", type=str, required=True, help="path to ground truth directory")
    parser.add_argument("--debugging", action='store_true', help="if set, save the internal output to out folder")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=False, help="output directory")
    args = parser.parse_args()
    return args


def sequence_inference(first_image: dict, 
                       first_image_cv2: np.array,
                       rest_images: List[dict], 
                       image_directory: str, 
                       image_files: List[str], 
                       sam_model: SamPredictor, 
                       mast3r_model: AsymmetricMASt3R, 
                       centers_first_img: np.array,
                       mask_size: np.array,
                       device: str, 
                       gts: np.array,
                       df: pd.DataFrame,
                       debugging=False,) -> pd.DataFrame: 
    
    # Process each image in the sequence
    for i, image in enumerate(tqdm(rest_images)):
        # should not always be the first image, but we can fix this later
        output = inference([(first_image, image)], mast3r_model, device, batch_size=1, verbose=False)
        
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach() # This is the descriptor ViT output I think
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=4,
                                                       device=device, dist='dot', block_size=2 ** 13)
        
        # We need to filter out the invalid matches, by edge and by the size of the image        
        H0, W0 = first_image['true_shape'][0]#512,288
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
       
        H1, W1 = image['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
        
        # The output is the following: reconverd_coords.shape = [n,2] where n is the number of matches found
        
        recovered_coords_0 = recover_coordinates(matches_im0, H0, W0, ori_H, ori_W)
        recovered_coords_1 = recover_coordinates(matches_im1, H0, W0, ori_H, ori_W)
        
        coordinates_0, coordinates_1 = find_corresponding_neighbors(flatten_points=centers_first_img, 
                                                                    mask_sizes= mask_size, 
                                                                    recovered_coords_0=recovered_coords_0, 
                                                                    recovered_coords_1=recovered_coords_1)
        
        # Load the current image
        
        current_image_path = os.path.join(image_directory, image_files[i + 1])
        current_image_cv2 = cv2.imread(current_image_path)
        current_image_cv2 = cv2.cvtColor(current_image_cv2, cv2.COLOR_BGR2RGB)   # or reload from disk at original size

        final_sam_masks = prompt_sam_with_mask_points(
            current_image_cv2, sam_model, coordinates_1, multimask_output=False
        )
         
        if debugging:

            # Save the visualization of the correspondences
            save_path = os.path.join(output_dir, f"correspondence/{i}.png")
            visualize_correspondences(first_image_cv2, current_image_cv2, np.vstack(coordinates_0), np.vstack(coordinates_1), save_path=save_path, points_correlation=-1)
            out_path = f"{output_dir}/masks/{i}.png"
            show_anns(current_image_cv2, final_sam_masks, alpha=0.35, figure_location=out_path)
            
        ious, f1, precision, recall = calculateMetrics(gts[i+1],final_sam_masks) # for each images
        
        df.loc[len(df)] = {'Name': image_files[i+1], 'IOU': ious, 'F1': f1, 'Precision': precision, 'Recall': recall}
        

    # 1. Compute mean of numeric columns:
    mean_values = df[['IOU', 'F1', 'Precision', 'Recall']].mean()

    # 2. Create a dictionary for the new row:
    mean_row = {
        'Name': 'MEAN',  # or any label you want
        'IOU': mean_values['IOU'],
        'F1': mean_values['F1'],
        'Precision': mean_values['Precision'],
        'Recall': mean_values['Recall']
    }
    df.loc[len(df)] = mean_row
    
    return df

def load_dataset(
    image_directory: str,
    gt_directory: str,
    sam_auto_mask_generator: SamAutomaticMaskGenerator,
):
    """
    Args:
        image_directory: Path to a folder containing images.
        gt_directory: Path to a folder containing ground truth annotations.
        sam_auto_mask_generator: An instance of SamAutomaticMaskGenerator.
    Returns:
        first_image: The first image in the sequence.
        rest_images: The rest of the images in the sequence.
        image_files: The names of the images in the sequence.
        gts: The ground truth annotations for the images in the sequence.
        ori_H: The original height of the first image.
        ori_W: The original width of the first image.
        mask_size: The size of the masks generated by the SAM model.
        centers_first_img: The centers of the masks in the first image.
        first_image_cv2: The first image in the sequence as a numpy array.
    """
    gt_annotations = image_directory.split('/')[-2]+'.npz'
    gt_annotations = os.path.join(gt_directory, gt_annotations)
    gts = np.array(list(np.load(gt_annotations).values())) ########### Need to be modified
    
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))])
    
    first_image_path = os.path.join(image_directory, image_files[0])
    
    first_image = load_images([first_image_path], size=512)[0]
    rest_images = load_images(image_directory, size=512)[1:]

    first_image_cv2 = cv2.imread(first_image_path)
    first_image_cv2 = cv2.cvtColor(first_image_cv2, cv2.COLOR_BGR2RGB)
    
    ori_H, ori_W = first_image_cv2.shape[:2]
    
    masks = sam_auto_mask_generator.generate(first_image_cv2)
    masks = merge_masks(masks, threshold=0.5)
    mask_size, centers_first_img = distribute_points_among_masks(masks) # in the shape of H,W
    
    return first_image, rest_images, image_files, gts, ori_H, ori_W, mask_size, centers_first_img, first_image_cv2


if __name__ == '__main__':
    
    args = parser()
    
    mast3r_model_path: str =  args.mast3r_checkpoint
    sam_version = args.sam_version
    sam_checkpoints = args.sam_checkpoint
    debugging = args.debugging
    output_dir = args.output_dir
    gt_dir = args.gt # gt directories
    image_dirs = args.image_dir
    
    if debugging:
        os.makedirs(output_dir, exist_ok=True)
        print("Debugging mode is on. The internal outputs will be saved to the 'outputs' folder.")


    
    ################################# Model Loading #################################
    device = "cuda"
    model = AsymmetricMASt3R.from_pretrained(mast3r_model_path).to(device)
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoints)
    predictor = SamPredictor(sam)
    model_type = "vit_h"

    sam.to(device=device)

    sam_automatic = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=5,
        pred_iou_thresh=0.90,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=9000,
    )
    
    for i in range(len(image_dirs)):
        image_directory = image_dirs[i]
        
        ################################# Data Loading #################################
        first_image, rest_images, image_files, gts, ori_H, ori_W, mask_size, centers_first_img, first_image_cv2 = load_dataset(
            image_directory, gt_dir, sam_automatic
        )

        ################################# Inference #################################

        df = pd.DataFrame(columns=['Name', 'IOU', 'F1', 'Precision', 'Recall'])
        df = sequence_inference(first_image, 
                                first_image_cv2, 
                                rest_images, 
                                image_directory, 
                                image_files, 
                                predictor, 
                                model, 
                                centers_first_img,
                                mask_size,
                                device, gts, df, debugging)    
    
        name = image_directory.split('/')[-1]
        df.to_csv(f'{output_dir}/{name}_results.csv', index=False)