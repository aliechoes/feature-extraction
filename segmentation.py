import numpy as np
from scipy.stats import ttest_ind
from skimage.filters import threshold_triangle
from skimage.filters import sobel
from skimage.morphology import disk,  remove_small_objects, binary_closing
from skimage.feature import greycomatrix, greycoprops 
from scipy.ndimage import binary_fill_holes

__all__ = [ 'segmentation_sobel',
            'segmentation_threshold']

def segmentation_sobel(image):
    """calculates the segmentation per channel using edge detection
    
    It first calculates the sobel filtered image to calculate the edges.
    Then removes the small objects, closes the binary shapes and 
    finally fills the shapes.

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    segmented_image :  3D array, shape (M, N, C)
        Segmentation of each channel of the input image.

    Raises
    -------
    None

    References
    -------
    ..  [1] http://jkimmel.net/so-you-want-to-segment-a-cell/

    Notes
    -----
    1.  It works best for brightfield channels in Imaging Flow Cytometry (IFC)
    2.  We have used triangle thresholding instead of otsu as it gives normally a bigger area of segmentation.
        one has to check whether it needs more thinnening or not.

    """
    segmented_image = image.copy()*0
    for ch in range(image.shape[2]):
        # calculate edges
        edges = sobel(image[:,:,ch])

        # segmentation
        threshold_level = threshold_triangle(edges)
        bw = edges > threshold_level # bw is a standard variable name for binary images
        
        # postprocessing
        bw_cleared = remove_small_objects(bw, 100) # clear objects <100 px
        # close the edges of the outline with morphological closing
        bw_close = binary_closing(bw_cleared, selem=disk(5))
        segmented_image[:,:,ch] = binary_fill_holes(bw_close)
    return segmented_image

def segmentation_threshold(image):
    """calculates the segmentation per channel using direct thresholding
    
    It calcualtes the threshold using triangle thresholding.
    Then removes the small objects, closes the binary shapes and 
    finally fills the shapes.

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    segmented_image :  3D array, shape (M, N, C)
        Segmentation of each channel of the input image.

    Raises
    -------
    None

    References
    -------
    .. [1] https://scikit-image.org/docs/dev/auto_examples/applications/plot_human_mitosis.html

    Notes
    -----
    1.  It works best for florescent channels in Imaging Flow Cytometry (IFC).
    2.  We have used triangle thresholding instead of otsu as it gives normally a bigger area of segmentation.
        one has to check whether it needs more thinnening or not.

    """
    segmented_image = image.copy()*0
    for ch in range(image.shape[2]):
        # segmentation
        threshold_level = threshold_triangle(image[:,:,ch])
        bw = image[:,:,ch] > threshold_level # bw is a standard variable name for binary images
        
        # postprocessing
        bw_cleared = remove_small_objects(bw, 100) # clear objects <100 px
        # close the edges of the outline with morphological closing
        bw_close = binary_closing(bw_cleared, selem=disk(5))
        segmented_image[:,:,ch] = binary_fill_holes(bw_close)

    return segmented_image
