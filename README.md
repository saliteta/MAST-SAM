# MAST-SAM 
Try to solve the multiview Inconsistency Problem

## Stage I (Utilize the Pretrain Model Well)
- We track the points correspondance using mast3R, 
![Correspondance of two adjancent picture](assets/correspondance.png)

- We segment the image using SAM
![AutoSegmentation](assets/rawSegmentation.png)

## State II (Modify The Pretrain Model's output)
- Convert output masks as output mask points
![Mask Point](assets/mask_point.png)

- Trance the mask points to next picture 
![Traced SAM](assets/traced_result.png)

## Current Problem
- Do not always trace the first image, some time need to update SAM segmentation, and some time, need to update the reference picture
![Limitation](assets/far_lag.png)




## Running Example
```
OPENBLAS_NUM_THREADS=4 python SAM_automask_butian.py --image_dir /home/xiongbutian/workspace/davis2017-evaluation/DAVIS/JPEGImages/480p/bear/ --debuggin
```