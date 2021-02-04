import cv2
import torch
from torchvision import transforms
import numpy as np


def image_transformer(input_image, train=True):
    """
    Using torchvision.transforms, make PIL image to tensor image
    with normalizing and flipping augmentations
    """
    
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    if train:
        transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    transformed_image = transformer(input_image)
    
    return transformed_image



def image_processor(input_image):
    """
    This function is to process given image using opencv.
    """
    blur = cv2.GaussianBlur(input_image, (3, 3), 0)

    denoised =  cv2.fastNlMeansDenoising(blur, None, 25, 7, 35)
    #denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
    denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
    
    kernel_sharpen = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) 
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
    
    _, thr = cv2.threshold(sharpened, 150, 255, 0)
    
    return thr


