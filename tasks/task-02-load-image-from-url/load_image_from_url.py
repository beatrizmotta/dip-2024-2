import argparse
import cv2
import numpy as np
import requests

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###

    response = requests.get(url)

    if (response.status_code != 200):
        print(f'Erro ao tentar carregar a imagem: {response.status_code}')
        return None
    else:
        img_array = np.asarray(bytearray(response.content), dtype="uint8")
        flags = kwargs.get('flags', cv2.IMREAD_COLOR)
        image = cv2.imdecode(img_array, flags)


    ### END CODE HERE ###
    
    return image
