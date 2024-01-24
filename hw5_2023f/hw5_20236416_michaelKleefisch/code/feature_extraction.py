import cv2
import numpy as np


def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a number of grid points x feature_size matrix.
    """
    #print(img.shape) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    #print(img.shape) 
    if feature == 'HoG':
        # HoG parameters
        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64


        hog = cv2.HOGDescriptor(_winSize=win_size, _blockSize=block_size, _blockStride=block_stride,
                                _cellSize=cell_size, _nbins=nbins, _derivAperture=deriv_aperture,
                                _winSigma=win_sigma, _histogramNormType=histogram_norm_type, 
                                _L2HysThreshold=l2_hys_threshold, _gammaCorrection=gamma_correction,
                                _nlevels=nlevels)
        grid_size = 16
        half_win_size = 16
        results = []
        for i in range(half_win_size, img.shape[0]-half_win_size, grid_size):
            for j in range(half_win_size, img.shape[1]-half_win_size, grid_size):
                result = hog.compute(img=img[i-half_win_size:i+half_win_size,j-half_win_size:j+half_win_size])
                results.append(result)
        # `.shape[0]` do not have to be (and may not) 1500,
        # but `.shape[1]` should be 36.
        results = np.array(results)
        return results

    elif feature == 'SIFT':

        sift = cv2.SIFT_create()

        grid_size = 20
        half_win_size = 16
        coordinates = [(x, y) for x in range(0, img.shape[0], grid_size) for y in range(0, img.shape[0], grid_size)]
        kpt = cv2.KeyPoint_convert(coordinates, size=32)

        keys, desc = sift.compute(img, kpt)

        return desc
        # `.shape[0]` do not have to be (and may not) 1500,
        # but `.shape[1]` should be 128.



