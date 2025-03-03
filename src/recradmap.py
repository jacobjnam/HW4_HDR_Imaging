import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def plot_with_colorbar(img,title=""):
    """
    This function might come in handy to deal with colorbar issue (if it's an issue for you)

    args
        img: an image to plot
        title: a string of the title for the plot
    """
    
    ax = plt.gca()
    im = ax.imshow(img)

    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def get_log_radiance_map(rawImg, log_exposure_time, solveG):
    """
    This function will calculate the radiance map using the solved G function in the previous part and the pixel intensity of the input images.
    
    args
        rawImg: m*n*k matrix. m*n is the size of the input image, and k is the number of input images). Input image series in a certain color channel (R,G,B channel). 
        logexpTime: (k,) matrix, k is the number of input images. log expore time.
        solveG: A (256,1) matrix. Solved G function in the previous section for corresponding color channel. 
    
    Output
        recRadMap: m*n matrix, m*n is the size of input images. Recovered radiance map
    """

    m = rawImg.shape[0]
    n = rawImg.shape[1]
    k = rawImg.shape[3]
    res = np.zeros([m, n, 3])
    
    for ii in range(3):
        EE = np.zeros([m, n])
        for jj in range(k):
            t = rawImg[:, :, ii, jj].squeeze()
            EE = EE + solveG[t, ii] - log_exposure_time[jj]
        res[:, :, ii] = np.reshape(EE, (m, n))/k
        
    return res

def plotRadMap(log_radiance_map):
    """
    This function will plot the heat map for the recovered radiance map in the previous part. Please you Matplotlib function to complete this task.
    The plotted heatmap should be similar to what is shown in the document. It should include a colorbar, which indicate the value of radiances.
    Three plots needs to be generated for each channel. So you need to call this function three times. You don't need to return anything in this function.
    
    args
        recRadMap: Recovered radiance map
        title: A string. The title of plotted radiance map, which should indicate which channal this map is.
    """
    plt.subplot(1,3,1)
    title='Red Channel'
    plot_with_colorbar(log_radiance_map[:,:,0],title)
    plt.subplot(1,3,2)
    title='Greenn Channel'
    plot_with_colorbar(log_radiance_map[:,:,1],title)
    plt.subplot(1,3,3)
    title='Blue Channel'
    plot_with_colorbar(log_radiance_map[:,:,2],title)