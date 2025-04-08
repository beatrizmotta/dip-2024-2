# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    # Your implementation here
    def mse(A, B):
        return np.mean((A.astype(np.float32) - B.astype(np.float32)) ** 2)
    def psnr(i1, i2):
        return 20 * np.log10(255.0 / np.sqrt(mse(i1, i2)))
    
    def ssim(i1, i2):
        i1 = i1.astype(np.float32)
        i2 = i2.astype(np.float32)
        
        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2

        mu1 = np.mean(i1)
        mu2 = np.mean(i2)
        sigma1_sq = np.var(i1)
        sigma2_sq = np.var(i2)
        sigma12 = np.mean((i1 - mu1) * (i2 - mu2))

        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        return numerator / denominator

    def npcc(i1, i2):
        i1 = i1.astype(np.float32)
        i2 = i2.astype(np.float32)

        i1_mean = i1 - np.mean(i1)
        i2_mean = i2 - np.mean(i2)

        numerator = np.sum(i1_mean * i2_mean)
        denominator = np.sqrt(np.sum(i1_mean**2) * np.sum(i2_mean**2))

        if denominator == 0:
            return 0.0  
        
        return numerator / denominator


    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }

