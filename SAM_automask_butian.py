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


def parser():
    parser = argparse.ArgumentParser("SAM Encoder Test", add_help=True)
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, default="/home/xiongbutian/workspace/Foundation_Models/SAM/sam_vit_h_4b8939.pth", required=False, help="path to sam checkpoint file")
    parser.add_argument("--mast3r_checkpoint", type=str, default='/home/xiongbutian/workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', required=False, help="path to sam checkpoint file")
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--debugging", action='store_true', help="if set, save the internal output to out folder")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=False, help="output directory")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    
    args = parser()
    
    device: str = args.device
    mast3r_model_path: str =  args.mast3r_checkpoint
    sam_version = args.sam_version
    sam_checkpoints = args.sam_checkpoint
    debugging = args.debugging
    output_dir = args.output_dir
    
    if debugging:
        os.makedirs(output_dir, exist_ok=True)
        print("Debugging mode is on. The internal outputs will be saved to the 'outputs' folder.")

    # To Do The pair correlation instead of first picture to the rest pictures
    model = AsymmetricMASt3R.from_pretrained(mast3r_model_path).to(device)
    image_directory = args.image_dir
    
    


    # Load the first image
    first_image_path = os.path.join(image_directory,'00000.jpg') 
    first_image = load_images([first_image_path], size=512)[0]
    rest_images = load_images(image_directory, size=512)[1:]

    first_image_cv2 = cv2.imread(first_image_path)
    first_image_cv2 = cv2.cvtColor(first_image_cv2, cv2.COLOR_BGR2RGB)
    ori_H, ori_W = first_image_cv2.shape[:2]
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoints)
    predictor = SamPredictor(sam)
    model_type = "vit_h"
    device = "cuda"

    sam.to(device=device)

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=5,
        pred_iou_thresh=0.90,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=9000,
    )
    masks = mask_generator_2.generate(first_image_cv2)
    masks = merge_masks(masks, threshold=0.5)
    
    mask_size, centers_first_img = distribute_points_among_masks(masks) # in the shape of H,W
    
    

    # Process each image in the sequence
    for i, image in enumerate(tqdm(rest_images)):
        # should not always be the first image, but we can fix this later
        output = inference([(first_image, image)], model, device, batch_size=1, verbose=False)
        
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
        

            
        ## TO DO: Loading the CV2 version image (or convert the current image format image -> CV2 image)as what we have done in line 67 and line 68
        current_image_cv2 = cv2.imread(os.path.join(image_directory, f"{i+1:05d}.jpg"))
        current_image_cv2 = cv2.cvtColor(current_image_cv2, cv2.COLOR_BGR2RGB)   # or reload from disk at original size
        ## TO DO: Loading the CV2 version image (or convert the current image format image -> CV2 image)as what we have done in line 67 and line 68
         
        if debugging:

            # Save the visualization of the correspondences
            save_path = os.path.join(output_dir, f"correspondence/{i}.png")
            visualize_correspondences(first_image_cv2, current_image_cv2, np.vstack(coordinates_0), np.vstack(coordinates_1), save_path=save_path, points_correlation=-1)
            

        final_sam_masks = prompt_sam_with_mask_points(
            current_image_cv2, predictor, coordinates_1, multimask_output=False
        )
        ## TO DO: Visualize the Final Result, each image should be visualized
        out_path = f"{output_dir}/masks/{i}.png"
        show_anns(current_image_cv2, final_sam_masks, alpha=0.35, figure_location=out_path)
        ## TO DO: Visualize the Final Result, each image should be visualized
