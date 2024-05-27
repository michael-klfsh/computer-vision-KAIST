import cv2
import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import label


def get_interest_points(image, descriptor_window_image_width):
    w2 = int(descriptor_window_image_width/2)

    image = cv2.filter2D(src=image, ddepth=-1, kernel=1/9*np.ones((3,3)), borderType=cv2.BORDER_REFLECT)

    sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobel_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    I_x = cv2.filter2D(src=image, ddepth=-1, kernel=sobel_x, borderType=cv2.BORDER_REFLECT)
    I_y = cv2.filter2D(src=image, ddepth=-1, kernel=sobel_y, borderType=cv2.BORDER_REFLECT)

    I_xx = cv2.filter2D(src=I_x, ddepth=-1, kernel=sobel_x, borderType=cv2.BORDER_REFLECT)
    I_yy = cv2.filter2D(src=I_y, ddepth=-1, kernel=sobel_y, borderType=cv2.BORDER_REFLECT)
    I_xy = cv2.filter2D(src=I_x, ddepth=-1, kernel=sobel_y, borderType=cv2.BORDER_REFLECT)

    I_xx = cv2.GaussianBlur(src=I_xx, ksize=(3,3), sigmaX=3)
    I_yy = cv2.GaussianBlur(src=I_yy, ksize=(3,3), sigmaX=3)
    I_xy = cv2.GaussianBlur(src=I_xy, ksize=(3,3), sigmaX=3)

    alpha = 0.04

    det = I_xx*I_yy - np.power(I_xy,2)
    trace = I_xx + I_yy

    C = det - alpha * trace**2
    result = np.zeros(C.shape)
    for x in range(w2,C.shape[0]-w2):
        for y in range(w2,C.shape[1]-w2):
            if C[x,y] > np.array([C[x-1,y],C[x+1,y],C[x,y-1],C[x,y+1],C[x+1,y+1],C[x+1,y-1],C[x-1,y+1],C[x-1,y-1]]).max():
                if C[x,y] > 2:
                    result[x,y] = 1
                else:
                    C[x,y] = 0
            else:
                C[x,y] = 0

    points = np.argwhere(result != 0)

    return points[:, 1], points[:, 0]

    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of interest points for the input image

    # 'image' can be grayscale or color, your choice.
    # 'descriptor_window_image_width', in pixels.
    #   This is the local feature descriptor width. It might be useful in this function to
    #   (a) suppress boundary interest points (where a feature wouldn't fit entirely in the image, anyway), or
    #   (b) scale the image filters being used.
    # Or you can ignore it.

    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.

    # Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    # If you're finding spurious interest point detections near the boundaries,
    # it is safe to simply suppress the gradients / corners near the edges of
    # the image.


    # After computing interest points, here's roughly how many we return
    # For each of the three image pairs
    # - Notre Dame: ~1300 and ~1700
    # - Mount Rushmore: ~3500 and ~4500
    # - Episcopal Gaudi: ~1000 and ~9000



def get_descriptors(image, x, y, descriptor_window_image_width):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of feature descriptors for a given set of interest points.

    # 'image' can be grayscale or color, your choice.
    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
    #   The local features should be centered at x and y.
    # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
    #   You can assume that descriptor_window_image_width will be a multiple of 4
    #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
    # If you want to detect and describe features at multiple scales or
    # particular orientations, then you can add input arguments.

    # 'features' is the array of computed features. It should have the
    #   following size: [length(x) x feature dimensionality] (e.g. 128 for
    #   standard SIFT)

    #Just use the patch as features
    # features = np.zeros((x.shape[0], 256))
    # for i in range(x.shape[0]):
    #     p_x = x[i]
    #     p_y = y[i]
    #     w2 = int(descriptor_window_image_width/2)
    #     w_left = int(p_x - w2)
    #     w_right = int(p_x + w2)
    #     w_up = int(p_y - w2)
    #     w_down = int(p_y + w2)
    #     #feature = np.zeros((256))
    #     square = image[w_up:w_down,w_left: w_right]
    #     features[i,:] = square.reshape((256))
    # return features



    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    n = x.shape[0]
    img_rows, img_cols = image.shape
    w2 = int(descriptor_window_image_width/2)
    features = np.zeros((n,128))
    for i in range(n):
        p_x = x[i]
        p_y = y[i]
        w_left = int(p_x - w2)
        w_right = int(p_x + w2)
        w_up = int(p_y - w2)
        w_down = int(p_y + w2)
        feature = np.zeros((128))
        if(w_left >=0 and w_right<img_cols and w_up>=0 and w_down<img_rows):
            square_grad_x = gradient_x[w_up:w_down,w_left: w_right]
            square_grad_y = gradient_y[w_up:w_down,w_left: w_right]
            grad_magnitudes, grad_angles = compute_gradients(square_grad_x, square_grad_y)

            for row in range(0, descriptor_window_image_width, 4):  
                for col in range(0, descriptor_window_image_width, 4):
                    grad_mag_small_square = grad_magnitudes[row:row+4, col:col+4]
                    grad_angles_small_square = grad_angles[row:row+4, col:col+4]
                    bins = np.abs(grad_angles_small_square // 45)
                    for b in range(8):
                        sum = np.sum(grad_mag_small_square[bins == b])

                        feature[8*row + 2*col+b] = sum
        f_max = feature.max()
        f_min = feature.min()
        if(f_max != 0):
            feature = (feature - f_min)/(f_max-f_min)
        features[i,:] = feature

    return features

def compute_gradients(gradient_x, gradient_y):
    gradient_magnitudes = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_angles = np.arctan2(gradient_y, gradient_x) * 180 / np.pi  # Convert to degrees

    weighted_gradient_magnitudes = gradient_magnitudes

    return weighted_gradient_magnitudes, gradient_angles



def match_features(features1, features2):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Please implement the "nearest neighbor distance ratio test",
    # Equation 4.18 in Section 4.1.3 of Szeliski.
    #
    # Please assign a confidence, else the evaluation function will not work.
    #

    # This function does not need to be symmetric (e.g., it can produce
    # different numbers of matches depending on the order of the arguments).

    # Input:
    # 'features1' and 'features2' are the n x feature dimensionality matrices.
    #
    # Output:
    # 'matches' is a k x 2 matrix, where k is the number of matches. The first
    #   column is an index in features1, the second column is an index in features2.
    #
    # 'confidences' is a k x 1 matrix with a real valued confidence for every match.
    n1 = features1.shape[0]
    n2 = features2.shape[0]

    nearest_index = np.zeros((n1, 2), dtype=int)
    nearest_distance = np.zeros((n1, 2))

    for i in range(n1):
        F1 = np.tile(features1[i,:], (n2,1))
        diff = F1 - features2
        distance = np.sqrt(np.sum(diff**2, 1))
        nearest_index[i,:] = [i, np.argsort(distance)[0]]

        nearest_distance[i,0] = distance[nearest_index[i,1]]
        nearest_distance[i,1] = distance[np.argsort(distance)[1]]

    confidences = np.zeros((n1))
    for i in range(n1):
        if(nearest_distance[i,1] != 0):
            confidences[i] = 1 - (nearest_distance[i,0] / nearest_distance[i,1])
    matches = nearest_index

    return matches, confidences
