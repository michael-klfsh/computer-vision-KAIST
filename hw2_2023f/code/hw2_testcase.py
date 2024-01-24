import cv2
import numpy as np
import time
import os
from my_filter2D import my_filter2D
from vis_hybrid_image import vis_hybrid_image
import math
import time
import matplotlib.pyplot as plt


def hw2_testcase():
    # This script has test cases to help you test your my_filter2D() function. You should verify here that your
    # output is reasonable before using your my_filter2D to construct a hybrid image in hw2.py. The outputs are all
    # saved and you can include them in your writeup. You can add calls to cv2.filter2D() if you want to check that
    # my_filter2D() is doing something similar.
    #
    # Revised by Dahyun Kang and originally written by James Hays.

    ## Setup
    name = 'submarine'
    border = 'ZEROS'
    test_image = cv2.imread('../data/'+name+'.bmp', -1) / 255.0
    test_image = cv2.resize(test_image, dsize=None, fx=0.7, fy=0.7, )

    result_dir = '../result/test'
    os.makedirs(result_dir, exist_ok=True)

    cv2.imshow('test_image', test_image)
    cv2.waitKey(10)

    ##################################
    ## Identify filter
    # This filter should do nothing regardless of the padding method you use.
    identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    identity_image = my_filter2D(test_image, identity_filter, bordertype=border)

    cv2.imshow('identity_image', identity_image)
    cv2.imwrite(os.path.join(result_dir, 'identity_image_'+name+'.jpg'), identity_image * 255)

    ##################################
    ## Small blur with a box filter
    # This filter should remove some high frequencies
    blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    blur_filter = blur_filter / sum(sum(blur_filter))  # making the filter sum to 1

    blur_image = my_filter2D(test_image, blur_filter, bordertype=border)

    cv2.imshow('blur_image', blur_image)
    cv2.imwrite(os.path.join(result_dir, 'blur_image_'+name+'.jpg'), blur_image * 255)

    ##################################
    ## Large blur
    # This blur would be slow to do directly, so we instead use the fact that Gaussian blurs are separable and blur
    # sequentially in each direction.
    large_1d_blur_filter = cv2.getGaussianKernel(25, 10)

    start_time = time.time()
    large_blur_image = my_filter2D(test_image, large_1d_blur_filter, bordertype=border)
    large_blur_image = my_filter2D(large_blur_image, large_1d_blur_filter.T, bordertype=border)  # notice the transpose operator
    print(f'[large_blur_image] time spent: {time.time() - start_time:.4} sec')

    cv2.imshow('large_blur_image', large_blur_image)
    cv2.imwrite(os.path.join(result_dir, 'large_blur_image_'+name+'.jpg'), large_blur_image * 255)

    # # If you want to see how slow this would be to do naively, try out this equivalent operation:
    large_blur_filter_naive = large_1d_blur_filter * large_1d_blur_filter.T
    
    start_time = time.time()
    large_blur_image_naive = my_filter2D(test_image, large_blur_filter_naive, bordertype=border)
    print(f'[large_blur_image_naive] time spent: {time.time() - start_time:.4} sec')
    
    cv2.imshow('large_blur_image_naive', large_blur_image_naive)
    cv2.imwrite(os.path.join(result_dir, 'large_blur_image_naive_'+name+'.jpg'), large_blur_image_naive * 255)

    ##################################
    ## Oriented filter (Sobel Operator)
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # should respond to horizontal gradients

    sobel_image = my_filter2D(test_image, sobel_filter, bordertype=border)

    # 0.5 added because the output image is centered around zero otherwise and mostly black
    cv2.imshow('sobel_image + 0.5', sobel_image + 0.5)
    cv2.imwrite(os.path.join(result_dir, 'sobel_image_'+name+'.jpg'), (sobel_image + 0.5) * 255)

    ##################################
    ## High pass filter (Discrete Laplacian)
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    laplacian_image = my_filter2D(test_image, laplacian_filter, bordertype=border)

    # 0.5 added because the output image is centered around zero otherwise and mostly black
    cv2.imshow('laplacian_image + 0.5', laplacian_image + 0.5)
    cv2.imwrite(os.path.join(result_dir, 'laplacian_image_'+name+'.jpg'), (laplacian_image + 0.5) * 255)

    ##################################
    ## High pass "filter" alternative
    high_pass_image = test_image - blur_image  # simply subtract the low frequency content

    cv2.imshow('high_pass_image + 0.5', high_pass_image + 0.5)
    cv2.imwrite(os.path.join(result_dir, 'high_pass_image_'+name+'.jpg'), (high_pass_image + 0.5) * 255)

    ##################################
    ## Done
    print('Press any key ...')
    cv2.waitKey(0)

def myOwnTest():
    test_image = cv2.imread('../data/cat.bmp', -1) / 255.0
    test_image = cv2.resize(test_image, dsize=None, fx=0.7, fy=0.7, )

    result_dir = '../result/test'
    os.makedirs(result_dir, exist_ok=True)

    Ownfilter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    Ownfilter = Ownfilter / sum(sum(Ownfilter))  # making the filter sum to 1

    image = my_filter2D(test_image, Ownfilter)
    cv2.imwrite(os.path.join(result_dir, 'ownTest_cat_normal.jpg'), image * 255)

def task4():
    pixelRange = [250000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000]
    dimRange = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    results = np.zeros((len(pixelRange), len(dimRange)))

    image = cv2.imread('../questions/RISDance.jpg', -1) / 255.0
    ratio = image.shape[1]/image.shape[0]

    for pixel in range(len(pixelRange)):
        for dim in range(len(dimRange)):
            blur_filter = np.ones((dimRange[dim],dimRange[dim]))
            blur_filter = blur_filter / sum(sum(blur_filter)) 

            newY = int(math.sqrt(pixelRange[pixel] * ratio))
            newX = int(newY / ratio)
            cur_image = cv2.resize(image, dsize=(newX,newY))
            startTime = time.time()
            cv2.filter2D(src=cur_image, dst=cur_image, ddepth=-1, kernel=blur_filter)
            results[pixel][dim] = time.time() - startTime

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(dimRange, pixelRange, results, 50, cmap='coolwarm')
    ax.set_xlabel('kernel dim')
    ax.set_ylabel('image res')
    ax.set_zlabel('time')
    plt.savefig('time_results.png')


if __name__ == '__main__':
    #hw2_testcase()
    #myOwnTest()
    task4()
