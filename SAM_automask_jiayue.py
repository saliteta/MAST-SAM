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
from utils.visualization import visualize_correspondences, show_anns, visualize_masks
from utils.points import recover_coordinates, prompt_sam_with_mask, find_corresponding_neighbors

from evaluation import calculateMetrics
import time

def parser():
    parser = argparse.ArgumentParser("SAM Encoder Test", add_help=True)
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, default="/home/xiongbutian/workspace/Foundation_Models/SAM/sam_vit_h_4b8939.pth", required=False, help="path to sam checkpoint file")
    parser.add_argument("--mast3r_checkpoint", type=str, default='/home/xiongbutian/workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', required=False, help="path to sam checkpoint file")
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--debugging", action='store_true', help="if set, save the internal output to out folder")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs2", required=False, help="output directory")
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
        os.makedirs(os.path.join(output_dir, "correspondence"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "confs"), exist_ok=True)
        print("Debugging mode is on. The internal outputs will be saved to the 'outputs' folder.")

    # To Do The pair correlation instead of first picture to the rest pictures
    model = AsymmetricMASt3R.from_pretrained(mast3r_model_path).to(device)
    image_directory = args.image_dir
    
    


    # Load the first image
    images = load_images(image_directory, size=512)
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    cv2_first_image = cv2.imread(os.path.join(image_directory, image_files[0]))
    ori_H, ori_W, _ = cv2_first_image.shape
    
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoints)
    predictor = SamPredictor(sam)
    model_type = "vit_h"
    device = "cuda"

    sam.to(device=device)

    
    # transform the dic GT to the array GT
    gt_dir = '/data/butian/sc_latent_sam/Annotations/Davis/' + image_directory.split('/')[-2] + '.npz'
    gtss = np.load(gt_dir)
    gts = np.array(list(gtss.values()))
    IOUS = list()
    F1 = list()
    PRECISION = list()
    RECALL = list() 
    

    # Process each image pair
    previous_mask = None
    first_image_cv2 = None
    for i, image in enumerate(tqdm(images)):
        if i == len(images) - 1:
            break
        # should not always be the first image, but we can fix this later
        output = inference([(image, images[i+1])], model, device, batch_size=1, verbose=False)
        
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach() # This is the descriptor ViT output I think
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=4,
                                                       device=device, dist='dot', block_size=2 ** 13)
        
        # We need to filter out the invalid matches, by edge and by the size of the image        
        H0, W0 = image['true_shape'][0]#512,288
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
       
        H1, W1 = image['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
        
        # The output is the following: reconverd_coords.shape = [n,2] where n is the number of matches found
        recovered_coords_0 = recover_coordinates(matches_im0, H0, W0, ori_H, ori_W)
        recovered_coords_1 = recover_coordinates(matches_im1, H0, W0, ori_H, ori_W)
        
        points_correspondance = np.array([recovered_coords_0, recovered_coords_1])
        
            
        # Load the current image
        current_image_path = os.path.join(image_directory, image_files[i + 1])
        current_image_cv2 = cv2.imread(current_image_path)
        current_image_cv2 = cv2.cvtColor(current_image_cv2, cv2.COLOR_BGR2RGB)   # or reload from disk at original size
        

        if i == 0:
            first_image_path = os.path.join(image_directory, image_files[i])
            first_image_cv2 = cv2.imread(first_image_path)
            first_image_cv2 = cv2.cvtColor(first_image_cv2, cv2.COLOR_BGR2RGB)

        final_sam_masks, points_correspondance_0, points_correspondance_1= prompt_sam_with_mask(
            image = current_image_cv2,  
            points_correspondace=points_correspondance,
            sam = sam,
            predictor=predictor, 
            previous_mask=previous_mask,
            first_image=first_image_cv2,
            update= (i%10 == 0),
        )
        
        
        previous_mask = final_sam_masks
        
        print(previous_mask.shape)
        
        
        #if debugging:
        if debugging:
            # Save the visualization of the correspondences
            corresponding_path = os.path.join(output_dir, f"correspondence/{i}.png")
            anns_path = f"{output_dir}/masks/{i}.png"
            conf_path = f"{output_dir}/confs/{i}.png"
            visualize_correspondences(first_image_cv2, current_image_cv2,
                                      np.vstack(points_correspondance_0), 
                                      np.vstack(points_correspondance_1), 
                                      save_path=corresponding_path, points_correlation=-1)
            visualize_masks(final_sam_masks.astype(bool), save_place=conf_path)
            show_anns(current_image_cv2, final_sam_masks, points_correspondance_1, gts[i+1], alpha=0.35, figure_location=anns_path)  
        
        first_image_cv2 = current_image_cv2
            
         
        ious, f1, precision, recall = calculateMetrics(gts[i+1],final_sam_masks)
        print(f"IOUS:{ious},F1:{f1},PRECISION:{precision},RECALL:{recall} for {i}.png")
    