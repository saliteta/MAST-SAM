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

- Trance the mask points to next picture (To Do)