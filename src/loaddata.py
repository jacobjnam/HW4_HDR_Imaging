import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
from os.path import join
from glob import glob

savedir = 'output/'

def save_fig_as_png(figtitle):
    '''
    Saves the current figure into the output folder
    The figtitle should not contain the ".png".
    This helper function shoudl be easy to use and should help you create the figures 
    needed for the report
    
    The directory where images are saved are taken from savedir in "Code.py" 
    and should be included in this function.
    
    Hint: The plt.gcf() might come in handy
    Hint 2: read about this to crop white borders
    https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    
    '''
    path = savedir + figtitle + '.png'
    plt.gcf()
    plt.savefig(path)

def loadImage(imgFolder):
    """
    This function will read the images and save all the input images into an array. The exposure times in seconds must be parsed from the filenames. 
    
    Input
        imgFolder(str): The location of the folder of input images.
    Output
        rawImg(np.array) - uint8): m*n*3*k array. m, n is the size of each image, and k is the number of input images. 3 represents k images in three different channels.
        expTime(1D-array): k array of exposure times extracted from the filenames
        
    Please note that when using OpenCV to read input images, it will generate an m*n*3 array, and the sequence is BGR CHANNEL instead of RGB! 
    Hint: There is an openCV function to convert from BGR to RGB (google it for more information)
    
    """
    lst = []
    exp = []
    for img_name in glob(imgFolder + "//*.png"):
        exposure = float(img_name[len(imgFolder)-1:-4])
        exp.append(exposure)
        img = cv2.imread(img_name)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_frame = np.array(rgb_img, dtype='float32')
        lst.append(np_frame)
    
    rawImg = np.asarray(lst, dtype='uint8')
    rawImg = np.moveaxis(rawImg, 0, -1)
    expTime = np.asarray(exp)
    
    exp_sort_order = np.argsort(expTime)
    expTime = expTime[exp_sort_order]
    rawImg = rawImg[..., exp_sort_order]
    
    return rawImg, expTime
    

def loadExposureTime(imgFolder):
    """
    This function will load the exposure time using the name of input images within the folder, then calculate the logorithm of loaded exposuretime.
    The name of each image is the expusure time in nanosecond. 
    So before calculating the log exposure time, please read the image, convert it to number from string, then convert it into second (1 second = 1e9 nanosecond).
    Input
        imgFolder:  A string. The location of the folder of input images.
    Output 
        expTime:  An (k,) array. k is the number of images. The exposure time.
        logExpTime: just the log of expTime
    """
    
    raise NotImplementedError
        

def create_measured_Z_values(rawImg, numSample= 1000, low = 0, high = 245):
    num_elements = rawImg.shape[0]*rawImg.shape[1]
    flat = rawImg.reshape(num_elements, rawImg.shape[2], rawImg.shape[3])
    randomIndex = np.random.randint(0, high=num_elements-1, size=numSample)   
    zValues = flat[randomIndex,:,:]
    zValues = np.swapaxes(zValues, 1, 2)
    return zValues