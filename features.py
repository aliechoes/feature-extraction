import numpy as np
from scipy.stats import kurtosis, skew
from scipy.spatial import distance as dist
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import hog
from skimage.transform import resize
from skimage.transform import integral_image
from skimage.feature import hog
from segmentation import segmentation_sobel
from skimage.exposure import histogram
from skimage.measure import shannon_entropy
from skimage.measure import moments_hu, inertia_tensor, inertia_tensor_eigvals
from skimage.measure import moments
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from skimage.morphology import medial_axis, skeletonize
from skimage.measure import label, regionprops
from skimage.feature import daisy
from sklearn.cluster import KMeans

__all__ = [ 'cell_level_shape_features', 
            'skeleton_features',
            'daisy_features', 
            'clustering_features',  
            'basic_statistical_features', 
            'moments_features', 
            'haar_like_features', 
            'hog_features', 
            'histogram_features', 
            'glcm_features', 
            'cross_Channel_distance_features', 
            'cross_Channel_boolean_distance_features']

def cell_level_shape_features(mask):
    """calculates the set of cell-skeleton based features 
    
    Calculates medial axis of the segmented cell and calculates the length,
    maximum and minimum thickness of the skeleton

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.

    Returns
    -------
    features :  dict  
        dictionary including percentiles, moments and sum per channel

    Raises
    -------
    None

    References
    -------
    .. [1] https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops

    Notes
    -----
    None

    """
    parameters = [  "area",  "bbox_area", "convex_area", "eccentricity", 
                    "equivalent_diameter", "euler_number", "extent", "filled_area","major_axis_length", 
                    "minor_axis_length", "orientation", "perimeter", "solidity" ]
    # storing the feature values
    features = dict()
    for ch in range(mask.shape[2]):
        for region in regionprops(mask[:,:,ch]):
            # take regions with large enough areas
            for par in parameters:
                if region.area >= 100:
                    features["cell_level_" + par + "_Ch" + str(ch+1)] = region[par]
                else: 
                    features["cell_level_" + par + "_Ch" + str(ch+1)] = 0.
    return features


def skeleton_features(mask):
    """calculates the set of cell-skeleton based features 
    
    Calculates medial axis of the segmented cell and calculates the length,
    maximum and minimum thickness of the skeleton

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
    for ch in range(mask.shape[2]):
        # calculating the medial axis and distance on the skeleton
        skel, distance = medial_axis(mask[:,:,ch], return_distance=True)
        dist_on_skel = distance * skel

        # storing the features
        features["skeleton_length_Ch" + str(ch+1)] = skel.sum()
        features["skeleton_thickness_max_Ch" + str(ch+1)] = dist_on_skel.max()
        if dist_on_skel.max() > 0.:
            features["skeleton_thickness_min_Ch" + str(ch+1)] = dist_on_skel[dist_on_skel > 0.].min()
        else:
            features["skeleton_thickness_min_Ch" + str(ch+1)] = 0.

    return features


def daisy_features(image):
    """calculates the set of cell-skeleton based features 
    
    Calculates medial axis of the segmented cell and calculates the length,
    maximum and minimum thickness of the skeleton

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
    
    # calculating the pixels per cells 
    for ch in range(image.shape[2]):
        temp_image = resize(image[:,:,ch].copy(), (32,32))
        daisy_features = daisy(temp_image, step=4, radius=9).reshape(1, -1)
        for i in range(daisy_features.shape[1]):
            features["daisy_" + str(i) + "_Ch" + str(ch+1)] = daisy_features[0][i]

    return features


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

        features["shannon_entropy_Ch" + str(ch+1)] = shannon_entropy(image[:,:,ch])
    
    return features

def clustering_features(image, k = 10):
    """calculates the centers of clusters per channel
    
    Calculates the centers of the clusters per channel using kmeans

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.
        
    k : int
        number of clusters

    Returns
    -------
    features :  dict  
        dictionary including center of the clusters per channel

    """    
    # storing the feature values
    features = dict()
    for ch in range(image.shape[2]):
        temp_image = image[:,:,ch].copy().reshape(image.shape[0]*image.shape[1],1  )
        kmeans = KMeans(n_clusters= k, random_state= 314).fit(temp_image)
        clusters = kmeans.cluster_centers_.tolist()
        for i in range(k):
            features["cluster_" + str(i) + "_Ch" + str(ch+1)] = clusters[i][0]

    return features

def moments_features(image):
    """calculates the set of moments for each channel
    
    Calculates the intertia tensor, intertia tensor eigenvalues, as well as 
    the moments of the image (https://en.wikipedia.org/wiki/Image_moment)

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
        hu_moments = moments_hu(image[:,:,ch]) 
        for i in range(len(hu_moments)):
            features["moments_hu_" + str(i+1) + "_Ch" + str(ch+1)] = hu_moments[i]
        
        inertia_tensor_calculated = inertia_tensor(image[:,:,ch]).ravel()
        features["inertia_tensor_1_Ch" + str(ch+1)] = inertia_tensor_calculated[0]
        features["inertia_tensor_2_Ch" + str(ch+1)] = inertia_tensor_calculated[1]
        features["inertia_tensor_3_Ch" + str(ch+1)] = inertia_tensor_calculated[3]
        
        inertia_tensor_eigvalues = inertia_tensor_eigvals(image[:,:,ch])
        features["inertia_tensor_eigvalues_1_Ch" + str(ch+1)] = inertia_tensor_eigvalues[0]
        features["inertia_tensor_eigvalues_2_Ch" + str(ch+1)] = inertia_tensor_eigvalues[1]   

        the_moments = moments(image[:,:,ch], order=5).ravel()

        for i in range(len(the_moments)):
            features["moments_" + str(i+1) + "_Ch" + str(ch+1)] = the_moments[i]
    
    return features

def haar_like_features(image):
    """calculates the set of haar-like features 
    
    Calculates the haar-like per channel. It first reshape the image to 64*64*C and
    then calcualtes the ['type-2-x', 'type-2-y'] features.
    For more info please refer to:
    https://scikit-image.org/docs/dev/auto_examples/applications/plot_haar_extraction_selection_classification.html

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.

    Returns
    -------
    features :  dict  
        dictionary including haar_1_Ch1, haar_2_Ch1 ...

    """
    # storing the feature values
    features = dict()
    for ch in range(image.shape[2]):
        temp_image = resize(image[:,:,ch].copy(), (32,32))
        ii = integral_image(temp_image)
        haar_fatures = haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1], ['type-2-x', 'type-2-y'])
        for i in range(len(haar_fatures)):
            features["haar_" + str(i+1) + "_Ch" + str(ch+1)] = haar_fatures[i]
    
    return features

def hog_features(image):
    """calculates the set of hog features 
    
    Calculates the hog features with
    For more info please refer to:
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.

    Returns
    -------
    features :  dict  
        dictionary including hog_1, hog_2 ...

    """
    temp_image = resize(image.copy(), (64,64))
    # calculating the pixels per cells 
    hog_features= hog(temp_image, orientations=8, pixels_per_cell=(12, 12),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)
    
    features = dict()
    for i in range(len(hog_features)):
        features["hog_" + str(i)] = hog_features[i]
    
    return features


def histogram_features(image, n_bins = 20):
    """calculates the histogram features per channel 
    
    Calculates the histograms for different channels
    For more info please refer to:
    https://scikit-image.org/docs/dev/api/skimage.exposure.html

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.
    n_bins : positive int 
        number of bins

    Returns
    -------
    features :  dict  
        dictionary including hist_0_Ch1, hist_1_Ch1 ...

    """  
 
    features = dict()
    for ch in range(image.shape[2]):
        hist, _ = np.histogram(image[:,:,ch], bins=n_bins)
        for i in range(n_bins):
            features["hist_" + str(i) + "_Ch" + str(ch+1)] = hist[i]
    
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

def cross_Channel_distance_features(image):
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
            features["manders_overlap_coefficient" + suffix] = (channel1.sum()*channel2.sum())/(np.power(channel1,2).sum()*np.power(channel2,2).sum())
            features["intensity_correlation_quotient" + suffix] = ((channel1>channel1.mean())*(channel2>channel2.mean())).sum()/(channel1.shape[0]) - 0.5 
            
            
    return features

def cross_Channel_boolean_distance_features(mask):
    """calculates the cross channel distance features 
    
    Calculates the distances across channels 

    Parameters
    ----------
    mask : 3D array, shape (M, N, C)
        The input mask with multiple channels. 

    Returns
    -------
    features :  dict  
        dictionary including different distances across channels

    """

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
