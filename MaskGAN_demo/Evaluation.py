import os
import cv2
import numpy as np
import pandas as pd
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

loss_fn_alex = lpips.LPIPS(net='alex')  

def compute_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def compute_lpips(img1, img2):
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if img1.shape[1] != 3:
        img1 = img1.repeat(1, 3, 1, 1)
    if img2.shape[1] != 3:
        img2 = img2.repeat(1, 3, 1, 1)
    d = loss_fn_alex(img1, img2)
    return d.item()

def compute_mse(img1, img2):
    err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    return err

def compute_psnr(img1, img2):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return 100  # Perfect match
    pixel_max = 255.0
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr

def compute_l1(img1, img2):
    err = np.mean(np.abs(img1.astype("float") - img2.astype("float")))
    return err

def compare_folders(original_folder, results_folder, save_csv="evaluation_results.csv"):
    original_images = sorted(os.listdir(original_folder))
    result_images = sorted(os.listdir(results_folder))

    ssim_scores = []
    lpips_scores = []
    mse_scores = []
    psnr_scores = []
    l1_scores = []

    records = []

    for orig_img_name in original_images:
        orig_base = os.path.splitext(orig_img_name)[0]
        matching_results = [f for f in result_images if f.startswith(orig_base + "_") or f == orig_img_name]
        if not matching_results:
            print(f"No match found for {orig_img_name}")
            continue

        res_img_name = matching_results[0]

        orig_path = os.path.join(original_folder, orig_img_name)
        res_path = os.path.join(results_folder, res_img_name)

        if not os.path.isfile(orig_path) or not os.path.isfile(res_path):
            continue

        orig_img = cv2.imread(orig_path)
        res_img = cv2.imread(res_path)

        if orig_img is None or res_img is None:
            print(f"Error loading {orig_img_name} or {res_img_name}")
            continue

        if orig_img.shape != res_img.shape:
            res_img = cv2.resize(res_img, (orig_img.shape[1], orig_img.shape[0]))

        # Compute all evaluation metrics
        ssim_score = compute_ssim(orig_img, res_img)
        lpips_score = compute_lpips(orig_img, res_img)
        mse_score = compute_mse(orig_img, res_img)
        psnr_score = compute_psnr(orig_img, res_img)
        l1_score = compute_l1(orig_img, res_img)

        ssim_scores.append(ssim_score)
        lpips_scores.append(lpips_score)
        mse_scores.append(mse_score)
        psnr_scores.append(psnr_score)
        l1_scores.append(l1_score)

        records.append({
            "Original Image": orig_img_name,
            "Result Image": res_img_name,
            "SSIM": ssim_score,
            "LPIPS": lpips_score,
            "MSE": mse_score,
            "PSNR": psnr_score,
            "L1 Loss": l1_score
        })

        print(f"Compared {orig_img_name} vs {res_img_name}: SSIM={ssim_score:.4f}, LPIPS={lpips_score:.4f}, MSE={mse_score:.2f}, PSNR={psnr_score:.2f}, L1={l1_score:.2f}")

    if ssim_scores:
        print("\n=== Overall Averages ===")
        print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
        print(f"Average LPIPS: {np.mean(lpips_scores):.4f}")
        print(f"Average MSE: {np.mean(mse_scores):.4f}")
        print(f"Average PSNR: {np.mean(psnr_scores):.2f}")
        print(f"Average L1 Loss: {np.mean(l1_scores):.4f}")
    else:
        print("No images compared.")

if __name__ == "__main__":
    original_folder = "../Original images"  
    results_folder = "../results"             
    compare_folders(original_folder, results_folder)

