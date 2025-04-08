# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    # Your implementation here
    def translate(img, distance_shift=(30, 30)):
        tx, ty = distance_shift
        h, w = img.shape[:2]
        output = np.zeros_like(img)

        x_start = max(0, tx)
        x_end = min(w, w + tx)
        y_start = max(0, ty)
        y_end = min(h, h + ty)

        src_x_start = max(0, -tx)
        src_x_end = src_x_start + (x_end - x_start)
        src_y_start = max(0, -ty)
        src_y_end = src_y_start + (y_end - y_start)

        output[y_start:y_end, x_start:x_end] = img[src_y_start:src_y_end, src_x_start:src_x_end]
        return output

    def rotate_90deg_clockwise(img):
        rotate_img = np.rot90(img, k=1, axes=(1, 0))
        return rotate_img
    
    def stretch(img):
        scale_by = 1.5
        height, width = img.shape[:2]
        new_width = int(width * scale_by)

        output = np.zeros((height, new_width),dtype='u1')

        for y in range(height):
            for x in range(new_width):
                origin_x = int(x / scale_by)
                output[y, x] = img[y, origin_x]

        return output

    def mirror(img):
        return np.flip(img, axis=1)

    def distort(img,):
        k = 0.1
        height, width = img.shape[:2]
        center_x, center_y = width / 2, height / 2

        y, x = np.indices((height, width), dtype=np.float32)
        x_normalized = (x - center_x) / center_x
        y_normalized = (y - center_y) / center_y
        r = np.sqrt(x_normalized**2 + y_normalized**2)

        scale = np.where(r == 0, 1, (r * (1 + k * r**2))/2)
        x_distorted = x_normalized * scale
        y_distorted = y_normalized * scale

        x_mapped_back = (x_distorted * center_x + center_x).astype(np.float32)
        y_mapped_back = (y_distorted * center_y + center_y).astype(np.float32)

        x_mapped_back = np.clip(np.round(x_mapped_back).astype(np.int32), 0, width-1)
        y_mapped_back = np.clip(np.round(y_mapped_back).astype(np.int32), 0, height-1)

        output = img[y_mapped_back, x_mapped_back]

        return output


    return {
        "translated": translate(img),
        "rotated": rotate_90deg_clockwise(img),
        "stretched": stretch(img),
        "mirrored": mirror(img),
        "distorted": distort(img)
    }
        