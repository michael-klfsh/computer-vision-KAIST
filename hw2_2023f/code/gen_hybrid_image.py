import cv2
import numpy as np
from my_filter2D import my_filter2D
import os


def gen_hybrid_image(image1, image2, cutoff_frequency):
    # Inputs:
    # - image1 -> The image from which to take the low frequencies.
    # - image2 -> The image from which to take the high frequencies.
    # - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian blur that will remove high frequencies.
    #
    # Task:
    # - Use my_filter2D to create 'low_frequencies' and 'high_frequencies'.
    # - Combine them to create 'hybrid_image'.

    ########################################################################
    # Remove the high frequencies from image1 by blurring it.
    # The amount of blur that works best will vary with different image pairs.
    ########################################################################
    # 1-D Gaussian kernel
    kernel = cv2.getGaussianKernel(cutoff_frequency * 4 + 1, cutoff_frequency)
    # 2-D Guassian kernel
    kernel = np.matmul(kernel, kernel.T)
    low_frequencies = my_filter2D(image1, kernel)

    ########################################################################
    # Remove the low frequencies from image2.
    # The easiest way to do this is to subtract a blurred version of image2 from the original version of image2.
    # This will give you an image centered at zero with negative values.
    ########################################################################
    low_freq_image2 = my_filter2D(image2, kernel)
    high_frequencies = image2 - low_freq_image2

    ########################################################################
    # Combine the high frequencies and low frequencies
    ########################################################################
    hybrid_image = high_frequencies + low_frequencies

    return hybrid_image, low_frequencies, high_frequencies

result_dir = '../result/hybrid'
os.makedirs(result_dir, exist_ok=True)
test_image1 = cv2.imread('../data/einstein.bmp', -1) / 255.0
test_image1 = cv2.resize(test_image1, dsize=None, fx=0.7, fy=0.7, )
test_image2 = cv2.imread('../data/marilyn.bmp', -1) / 255.0
test_image2 = cv2.resize(test_image2, dsize=None, fx=0.7, fy=0.7, )

hybrid, low, high = gen_hybrid_image(test_image1,test_image2, 2)
cv2.imshow('hybrid_image', hybrid)
cv2.imshow('low_image', low)
cv2.imshow('high_image', high)
cv2.waitKey(0)
cv2.imwrite(os.path.join(result_dir, 'identity_image_catdog.jpg'), hybrid * 255)
