import numpy as np
from scipy.stats import kurtosis, skew
from scipy.spatial import distance as dist
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import hog
from skimage.transform import resize
from skimage.transform import integral_image
from segmentation import segmentation_sobel

def basic_statistical_features(image):
    """calculates the set of basic statistical features 
    
    Calculates the standard statistical features per channel every 10th percentile,
    sum of the pixel values and different moments

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.

    Returns
    -------
    features :  dict  
        dictionary including percentiles, moments and sum per channel 

    """  
    # storing the feature values
    features = dict()
    for ch in range(image.shape[2]):
        # percentiles
        features["min_intensity_Ch" + str(ch+1)] = image[:,:,ch].min()
        features["percentile10_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.1)
        features["percentile20_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.2)
        features["percentile30_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.3)
        features["percentile40_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.4)
        features["percentile50_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.5)
        features["percentile60_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.6)
        features["percentile70_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.7)
        features["percentile80_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.8)
        features["percentile90_intensity_Ch" + str(ch+1)] = np.percentile(image[:,:,ch] , 0.9)
        features["max_intensity_Ch" + str(ch+1)] = image[:,:,ch].max()

        # pixel sum
        features["total_intensity_Ch" + str(ch+1)] = image[:,:,ch].sum()

        # moments
        features["mean_intensity_Ch" + str(ch+1)] = image[:,:,ch].mean()
        features["std_intensity_Ch" + str(ch+1)] = image[:,:,ch].std()
        features["kurtosis_intensity_Ch" + str(ch+1)] = kurtosis(image[:,:,ch].ravel()) 
        features["skew_intensity_Ch" + str(ch+1)] = skew(image[:,:,ch].ravel()) 
    
    return features

def glcm_features(image):
    """calculates the glcm features 
    
    Calculates the features per channel using glcm features including
    contrast, dissimilarity, homogeneity, ASM, energy and correlation.
    For more info please refer to:
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    features :  dict  
        dictionary including 'contrast_Chx', 'dissimilarity_Chx', 'homogeneity_Chx'
        'ASM_Chx', 'energy_Chx' and 'correlation_Chx' per channel where 
        x will be substituted by the channel number starting from 1. 

    """
    features = dict()
    for ch in range(image.shape[2]):
        # create a 2D temp image 
        temp_image = image[:,:,ch].copy()
        temp_image = (temp_image/temp_image.max())*255 # use 8bit pixel values for GLCM
        temp_image = temp_image.astype('uint8') # convert to unsigned for GLCM

        # calculating glcm
        glcm = greycomatrix(temp_image,distances=[5],angles=[0],levels=256)

        # storing the glcm values
        features["contrast_Ch" + str(ch+1)] = greycoprops(glcm, prop='contrast')[0,0]
        features["dissimilarity_Ch" + str(ch+1)] = greycoprops(glcm, prop='dissimilarity')[0,0]
        features["homogeneity_Ch" + str(ch+1)] = greycoprops(glcm, prop='homogeneity')[0,0]
        features["ASM_Ch" + str(ch+1)] = greycoprops(glcm, prop='ASM')[0,0]
        features["energy_Ch" + str(ch+1)] = greycoprops(glcm, prop='energy')[0,0]
        features["correlation_Ch" + str(ch+1)] = greycoprops(glcm, prop='correlation')[0,0]

    return features

def cross_channel_distance_features(image):
    """calculates the cross channel distance features 
    
    Calculates the distances across channels 

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    features :  dict  
        dictionary including different distances across channels

    """
    features = dict()
    for ch1 in range(image.shape[2]):
        for ch2 in range(ch1+1,image.shape[2]):
            # rehaping the channels to 1D
            channel1 = image[:,:,ch1].ravel()
            channel2 = image[:,:,ch2].ravel()

            # creating the suffix name for better readability
            suffix = "_Ch" + str(ch1 + 1) + "_Ch" + str(ch2 + 1)

            # storing the distance values
            features["braycurtis_distance" + suffix] = dist.braycurtis(channel1,channel2)
            features["canberra_distance" + suffix] = dist.canberra(channel1,channel2)
            features["chebyshev_distance" + suffix] = dist.chebyshev(channel1,channel2)
            features["cityblock_distance" + suffix] = dist.cityblock(channel1,channel2)
            features["correlation_distance" + suffix] = dist.correlation(channel1,channel2)
            features["cosine_distance" + suffix] = dist.cosine(channel1,channel2)
            features["euclidean_distance" + suffix] = dist.euclidean(channel1,channel2)
            features["jensenshannon_distance" + suffix] = dist.jensenshannon(channel1,channel2)
            features["minkowski_distance" + suffix] = dist.minkowski(channel1,channel2)
            features["sqeuclidean_distance" + suffix] = dist.sqeuclidean(channel1,channel2)
    
    return features

def cross_channel_boolean_distance_features(image):
    """calculates the cross channel distance features 
    
    Calculates the distances across channels 

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    features :  dict  
        dictionary including different distances across channels

    """
    # create a channel-wise boolean mask from the image using segmentation
    mask = segmentation_sobel(image)

    features = dict()
    for ch1 in range(mask.shape[2]):
        for ch2 in range(ch1+1,mask.shape[2]):
            # rehaping the channels to 1D
            channel1 = mask[:,:,ch1].ravel()
            channel2 = mask[:,:,ch2].ravel()

            # creating the suffix name for better readability
            suffix = "_Ch" + str(ch1 + 1) + "_Ch" + str(ch2 + 1)

            # storing the distance values
            features["dice_distance" + suffix] = dist.dice(channel1,channel2)
            features["hamming_distance" + suffix] = dist.hamming(channel1,channel2)
            features["jaccard_distance" + suffix] = dist.jaccard(channel1,channel2)
            features["kulsinski_distance" + suffix] = dist.kulsinski(channel1,channel2)
            features["rogerstanimoto_distance" + suffix] = dist.rogerstanimoto(channel1,channel2)
            features["russellrao_distance" + suffix] = dist.russellrao(channel1,channel2)
            features["sokalmichener_distance" + suffix] = dist.sokalmichener(channel1,channel2)
            features["sokalsneath_distance" + suffix] = dist.sokalsneath(channel1,channel2)
            features["yule_distance" + suffix] = dist.yule(channel1,channel2)
    
    return features