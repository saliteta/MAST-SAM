# from time import time
# import argparse
# from skimage import io
# from sklearn.metrics import jaccard_score,f1_score,auc
# import numpy as np
# import pandas as pd
# from davis2017.evaluation import DAVISEvaluation
# from tqdm import tqdm
# # from skimage import io, color
# import pandas as pd
# import os
import numpy as np

# def parser():
#     parser = argparse.ArgumentParser(description="Calculate metrics for datasets.")
#     parser.add_argument('--gt_dir', type=str, help='Directory for ground truth npz files.')
#     parser.add_argument('--pred_dir', type=str, help='Directory for prediction npz files.')

def copyGT(gt, pred):
    copiedGT = np.zeros(pred.shape)
    B = pred.shape[0]
    for i in range(B):
        copiedGT[i,:,:] = gt
    return copiedGT


def calculateMetrics(gt, preds):
    max_iou = 0
    for i in range(np.array(preds).shape[0]):  
        mask = preds[i]
        
        intersection = np.sum(np.bitwise_and(gt, mask))  
        union = np.sum(np.bitwise_or(gt, mask))  
        
        iou = intersection / union if union != 0 else 0
        if iou > max_iou:
            max_iou = iou

            # calculate precision and recall
            true_positive = intersection
            predicted_positive = np.sum(mask)  # TP + FP
            actual_positive = np.sum(gt)      # TP + FN

            precision = true_positive / predicted_positive if predicted_positive != 0 else 0
            recall = true_positive / actual_positive if actual_positive != 0 else 0

            # calculate f-value
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    if max_iou == 0:
        precision = 0
        recall = 0
        f1 = 0
    return max_iou, f1, precision, recall

# def Cal_IOU(gt, preds):
#     #复制B个一样的GT
#     copiedGT = copyGT(gt, preds)
#     AND = np.bitwise_and(copiedGT, preds)
#     Union = np.bitwise_or(copiedGT, preds)
        
#     sum_AND = np.sum(AND, axis=(1, 2)) # True positive (B,)
#     sum_UNION = np.sum(Union, axis=(1,2))
#     sum_pred = np.sum(preds, axis=(1,2)) #TP + FP 
#     sum_value = np.sum(copiedGT, axis=(1,2)) #TP + FN
        
        
#     # 计算IoU
#     IOU = sum_AND / sum_UNION

#      # 计算Precision和Recall
#     precision = sum_AND / sum_pred  # 计算所有的Precision值
#     recall = sum_AND / sum_value      # 计算所有的Recall值


#     # 计算F1值
#     if precision is not None and recall is not None:
#         f1 = 2 * precision * recall / (precision + recall)
#     else:
#         f1 = 0  # 如果没有有效值则设为0

#     # 处理可能的NaN值
#     f1 = np.nan_to_num(f1)  # 将NaN替换为0
        

#     return IOU, f1, precision,recall

    
def update_csv(filename, data):
    
    new_row = pd.DataFrame([data])
    
    # 检查 CSV 文件是否存在
    if not os.path.exists(filename):
        # 如果不存在，创建新文件，并写入数据
        new_row.to_csv(filename, index=False)
    else:
        # 如果存在，读取原有数据，然后使用 concat 添加新行
        df = pd.read_csv(filename)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)
    
# args = parser()
# gt_dir = args.gt_dir
# pred_dir = args.pred_dir
def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for datasets.")
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory for ground truth npz file.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory for prediction npz file.')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file to update results.')
    args = parser.parse_args()

    IOU, F1, PRECISION, RECALL = Cal_IOU(args.gt_dir, args.pred_dir)
# gt_dir = '/home/xiongbutian/workspace/sc_latent_sam/Annotations/Davis/bear.npz'
# pred_dir = '/home/xiongbutian/workspace/sc_latent_sam/output_files/Davis-script-adjust-output/bear.npz'

# IOU, F1, PRECISION, RECALL = Cal_IOU(gt_dir, pred_dir)
    # print("Iou is",IOU)
    # print('F1 is:', F1)
    # print('Precision is:', PRECISION)
    # print('Recall is:', RECALL)
    # print("    ")
    data = {
        'Filename': os.path.basename(args.gt_dir),
        'IOU': IOU,
        'F1': F1,
        'Precision': PRECISION,
        'Recall': RECALL
    }

    # Update CSV
    update_csv(args.csv_file, data)
    print("Metrics updated in CSV file.")
if __name__ == "__main__":
    main()


    