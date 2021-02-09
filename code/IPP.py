# Image Pre Processing for Morphology Project
import os
import cv2
import numpy as np 

def get_images(inputs='../images/'):
    '''
    todo:
    ----- 
        resize input images to less than 256*X or X*256

    input:
    ------ 
        inputs - folder with input images (defaulted)
    
    output:
    -------
        im - Image data
        names of all images 
    ---
    '''
    im = []
    for im_name in os.listdir(inputs):
        im.append(cv2.imread(inputs+im_name))
    return np.asarray(im), os.listdir(inputs)

if __name__ == '__main__':
    get_image()