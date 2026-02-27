import cv2
import numpy as np
import os
from glob import glob

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Identical images
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Paths
enhanced_dir = r'C:\Users\aryan\Downloads\Zero-DCE-master\Zero-DCE_code\results'           # Folder where your enhanced images are saved
ground_truth_dir = r'C:\Users\aryan\Downloads\Zero-DCE-master\Zero-DCE_code\TRUTH'     # Folder with ground truth images

# List all image filenames
enhanced_files = sorted(glob(os.path.join(enhanced_dir, '*.png')))  # or .jpg
gt_files = sorted(glob(os.path.join(ground_truth_dir, '*.png')))

psnr_scores = []

for ef, gf in zip(enhanced_files, gt_files):
    enhanced = cv2.imread(ef)
    gt = cv2.imread(gf)

    # Resize if needed
    if enhanced.shape != gt.shape:
        gt = cv2.resize(gt, (enhanced.shape[1], enhanced.shape[0]))

    psnr = calculate_psnr(enhanced, gt)
    psnr_scores.append(psnr)
    print(f"PSNR for {os.path.basename(ef)}: {psnr:.2f} dB")

print(f"\nAverage PSNR: {np.mean(psnr_scores):.2f} dB")
