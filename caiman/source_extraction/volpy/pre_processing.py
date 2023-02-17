# -*- coding: utf-8 -*-
"""
Code for background correction

@author: Rui Silva
@date: 22/12/2022
"""
import numpy as np
import os
import tifffile
import logging
from scipy import interpolate
import caiman as cm

##########################################################################################
#    BACKGROUND FUNCTIONS
##########################################################################################

def moving_average(vector, n=3):
    """
    Moving average function. First n values of the vector are simply the original values
    
    Variables
        vector: vector on which to do moving average
        n: number of values to average out
    
    Returns:
        rolAvgd: rolling averaged vector
    """
    
    rolAvgd = np.cumsum(vector, dtype=float)
    rolAvgd[n:] = rolAvgd[n:] - rolAvgd[:-n]
    rolAvgd[n - 1:] = rolAvgd[n - 1:] / n
    rolAvgd[0:n] = np.mean(vector[0:n])
    
    return rolAvgd

def splineBackground(imgs,frames_per_knot = 500, normalize = False):
    """
    Remove photobleaching by spline correction
    
    Variables:
        imgs: image stack
        frames_per_knot: Number of frames per knot
        normalize: Normalize spline fit curve ((value-mean)/std)
        
    Returns:
        outputImg: Corrected image sequence
        yfit: Spline fit curve
    """
    
    #Get shape of the input images
    frames,height,width = imgs.shape
    
    #Get knots_number
    knot_numbers = int(frames/frames_per_knot)-1
    
    #Get x and y variables of the average of the whole image stack over the frames
    x = range(0,frames)
    y= np.mean(np.mean(imgs,axis=1),axis=1)
    y = moving_average(y,100) #Use a 100 frames high pass filter to remove effects of artifacts
    
    #Interpolate a spline that fits the values obtained above
    x_new = np.linspace(0, 1, knot_numbers+2)[1:-1]
    q_knots = np.quantile(x, x_new)
    t,c,k = interpolate.splrep(x, y, t=q_knots, s=1)
    yfit = interpolate.BSpline(t,c,k)(x)
    
    #Remove mean of the fitted curve. In this way the output result will only remove the photobleaching effect fluctuations
    yfit_corr = yfit - np.mean(yfit)
    
    #Detrend image with obtained spline
    outputImg = np.zeros(shape = imgs.shape,dtype=imgs.dtype)
    for i in range(0,frames):
        outputImg[i,:,:] = imgs[i,:,:]-yfit_corr[i]
        
    if (normalize is True):
        outputImg = (outputImg - np.mean(np.mean(outputImg,1),1))/np.std(np.std(outputImg,1),1)
        
    return outputImg,yfit

def median_trend_removal(imgs,window = 10):
    """
    Remove shot noise and motion correction effects
    
    Variables:
        imgs: image stack
        window: frames window for rolling average
        
    Returns:
        outputImg: Corrected image sequence
        medianVector: Median fit curve
    """
    
    #Turn the 3D image into a 2D image for faster computation of median
    outputImage = np.reshape(imgs.copy(), (imgs.shape[0], -1))
    
    medianVector = np.median(outputImage, 1) #Obtain median vector corresponding to the median value of each frame
    medianVector = moving_average(medianVector,n=window) #Use a 10 frames high pass filter to remove effects of artifacts
    
    outputImage = np.transpose(np.transpose(outputImage) - medianVector) #Remove median Vector for outputImage
    
    #Return outputImage as a 3D image again and sum up the mean value of the median Vector to return pixel values to the original range
    outputImage = np.reshape(outputImage,imgs.shape)+np.mean(medianVector) 
    
    return outputImage,medianVector

def voltage_preprocessing(fnames, saveDir = None, nbits = np.float16):
    """
    Remove photobleaching and shot noise effects
    
    Variables:
        filename: Name of the file of the file to analyze. Normally it is a motion corrected file.
        ds: Downscale factors
        saveDir: Directory to save the images. If None is given, the denoising results are saved in a folder
                folder called 'Denoise results' and the Denoised image is saved in the directory of that folder.
        saveAll: If True saves both intermediate steps as 3D images. If False, only saves the last denoised image.
        
    Returns:
        output_filename: Filename of the Denoised image
    """
    
    if isinstance(fnames,str):
        fnames = [fnames]
    
    if len(fnames) > 1:
        fnames = [fnames[0]]
    
    if os.path.exists(fnames[0]):
        
        logging.info("Background removal of  "+str(fnames[0]))
        #Open images as 16bit caiman movies. This is normally a tif file but in this way, it prevents errors if the file is not .tif
        mov = cm.base.movies.load(fnames)
        
        #Create folder for saving denoising results
        if saveDir == None:
            saveDir = os.path.dirname(fnames[0])
        folderName = os.path.join(saveDir, "Denoise results")
        
        if os.path.exists(folderName) == False:
            os.makedirs(folderName)
        
        #Remove photobleach effect
        mov,splineFit = splineBackground(mov)
        
        np.save(os.path.join(folderName, 'spline.npy'),splineFit)
        
        logging.info("Spline background removal done")
        
        #### Median detrending
        mov,medianVector = median_trend_removal(mov)
        
        np.save(os.path.join(folderName, 'medianTrend.npy'),medianVector)
        
        logging.info('Median detrending done')
        
        #Save denoised image
        output_filename = os.path.join(folderName, 'Denoised.tif')
        tifffile.imwrite(output_filename, mov.astype(nbits),
                          append = False,
                          bigtiff=True,
                          contiguous = True)
    else:
        logging.error('File: ' + fnames + ' does not exist!')
    
    return output_filename

def sucessive_preprocessing(fnames, saveDir=None):
    
    outputNames = fnames
    for i in range(0,len(fnames)):
        outputName = voltage_preprocessing(fnames[i],saveDir = saveDir)
        outputNames[i] = outputName
        
    return outputNames