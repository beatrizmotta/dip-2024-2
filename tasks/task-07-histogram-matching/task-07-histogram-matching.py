# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    # Your implementation here
    def get_gray_histogram(img):
        if (len(img.shape) == 3):
            raise Exception("Must be grayscale!")
  
        return cv.calcHist([img], [0], None, [256], [0, 256])
    
    def calc_cdf(histogram):
        cdf = histogram.cumsum()
        cdf_normalized = cdf / cdf[-1]
        return cdf_normalized

    def match_gray_histograms(source, reference):
        source_hist = get_gray_histogram(source)
        reference_hist = get_gray_histogram(reference)

        source_cdf = calc_cdf(source_hist)
        reference_cdf = calc_cdf(reference_hist)

        look_up_table = np.zeros(256, dtype=np.uint8)

        for value in range(256):
            diff = np.abs(reference_cdf - source_cdf[value])
            matched = np.argmin(diff)
            look_up_table[value] = matched

        matched_channel = cv.LUT(source, look_up_table)
        return matched_channel

    def match_rgb_histogram(source, reference):
        matched_channels = []

        for i in range(3):
            source_channel = source[:, :, i] 
            reference_channel = reference[:, :, i] 

            matched = match_gray_histograms(source_channel, reference_channel)
            matched_channels.append(matched)

        return cv.merge(matched_channels)

    output = match_rgb_histogram(source_img, reference_img)
    return output
