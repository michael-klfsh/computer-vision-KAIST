################################################################
# WARNING
# --------------------------------------------------------------
# When you submit your code, do NOT include blocking functions
# in this file, such as visualization functions (e.g., plt.show, cv2.imshow).
# You can use such visualization functions when you are working,
# but make sure it is commented or removed in your final submission.
#
# Before final submission, you can check your result by
# set "VISUALIZE = True" in "hw3_main.py" to check your results.
################################################################
from utils import normalize_points
from utils import homogenize
import numpy as np
import cv2
import multiprocessing
import functools



#=======================================================================================
# Your best hyperparameter findings here
WINDOW_SIZE = 9 #7
DISPARITY_RANGE = 40 #40
AGG_FILTER_SIZE = 15#5



#=======================================================================================
def bayer_to_rgb_bilinear(bayer_img):
    rgb_img = np.full((bayer_img.shape[0], bayer_img.shape[1], 3), -1)

    for i in range(bayer_img.shape[0]):
        for j in range(bayer_img.shape[1]):
            if(i%2 == 0 and j%2 == 0):      # red
                rgb_img[i,j,0] = bayer_img[i,j]
                rgb_img[i,j,1] = calc_green(bayer_img,i,j)
                rgb_img[i,j,2] = calc_red_blue(bayer_img,i,j)

            elif(i%2 == 1 and j%2 == 1):    # blue
                rgb_img[i,j,0] = calc_red_blue(bayer_img, i, j)
                rgb_img[i,j,1] = calc_green(bayer_img,i,j)
                rgb_img[i,j,2] = bayer_img[i,j]

            else:                           # green
                rgb_img[i,j,0] = calc_red_at_green(bayer_img, i, j)
                rgb_img[i,j,1] = bayer_img[i,j]
                rgb_img[i,j,2] = calc_blue_at_green(bayer_img,i,j)

    return rgb_img

def calc_green(bayer_img, i, j):
    sum = 0
    count = 0
    if(i > 0):
        sum += bayer_img[i-1,j]
        count += 1
    if(i < bayer_img.shape[0]-1):
        sum += bayer_img[i+1,j]
        count += 1
    if(j > 0):
        sum += bayer_img[i,j-1]
        count += 1
    if(j < bayer_img.shape[1]-1):
        sum += bayer_img[i,j+1]
        count += 1
    return sum/count

def calc_red_blue(bayer_img, i, j):
    sum = 0
    count = 0
    if(i > 0):
        if( j > 0):
            sum += bayer_img[i-1,j-1]
            count += 1
        if(j < bayer_img.shape[1]-1):
            sum += bayer_img[i-1, j+1]
            count += 1
    if(i < bayer_img.shape[0]-1):
        if( j > 0):
            sum += bayer_img[i+1,j-1]
            count += 1
        if(j < bayer_img.shape[1]-1):
            sum += bayer_img[i+1, j+1]
            count += 1
    return sum/count

def calc_red_at_green(bayer_img, i,j):
    sum = 0
    count = 0
    if(i%2==0):
        if(j>0):
            sum += bayer_img[i,j-1]
            count += 1
        if(j< bayer_img.shape[1]-1):
            sum += bayer_img[i,j+1]
            count += 1
    else:
        if(i>0):
            sum += bayer_img[i-1,j]
            count += 1
        if(i< bayer_img.shape[0]-1):
            sum += bayer_img[i+1,j]
            count += 1
    return sum/count

def calc_blue_at_green(bayer_img,i,j):
    sum = 0
    count = 0
    if(i%2==1):
        if(j>0):
            sum += bayer_img[i,j-1]
            count += 1
        if(j< bayer_img.shape[1]-1):
            sum += bayer_img[i,j+1]
            count += 1
    else:
        if(i>0):
            sum += bayer_img[i-1,j]
            count += 1
        if(i< bayer_img.shape[0]-1):
            sum += bayer_img[i+1,j]
            count += 1
    return sum/count

#=======================================================================================
def bayer_to_rgb_bicubic(bayer_img):
    # Your code here
    ################################################################
    rgb_img = None


    ################################################################
    return rgb_img



#=======================================================================================
def calculate_fundamental_matrix(pts1, pts2):
    # Assume input matching feature points have 2D coordinate
    assert pts1.shape[1]==2 and pts2.shape[1]==2
    # Number of matching feature points should be same
    assert pts1.shape[0]==pts2.shape[0]

    no_samples = pts1.shape[0]
    pts1_hom = homogenize(pts1)
    pts2_hom = homogenize(pts2)
    
    pts1_norm, T_1= normalize_points(pts1_hom.T, 2)
    pts2_norm, T_2= normalize_points(pts2_hom.T, 2)

    pts1 = pts1_norm.T
    pts2 = pts2_norm.T

    A = []

    for i in range(len(pts1)):
            p1 = pts1[i,:]
            p2 = pts2[i,:]
            a_i = np.outer(p2, p1).reshape((9))
            A = np.concatenate((A,a_i))
    A = A.reshape((no_samples,9))

    #U,S,V_T = np.linalg.svd(A)     #For some reason it is giving wrong signs
    
    eigval, eigvec = np.linalg.eig(A.T@A)
    #S_2 = S*S
    f= eigvec[:, np.argmin(eigval)]
    #f= V_T.T[:,np.argmin(S)]
    F = f.reshape((3,3))
    U,S,V_T = np.linalg.svd(F)
    S[-1] = 0
    F_new = U @ (S*np.identity(3)) @ V_T
    F = T_2.T @ F_new @ T_1

    fundamental_matrix = F
    return fundamental_matrix



#=======================================================================================
def transform_fundamental_matrix(F, h1, h2):
    F_new = h2.T@F@h1
    e1 = np.linalg.solve(F_new,np.linalg.solve(np.linalg.inv(h1), np.array([1,0,0]).T))
    e2 = np.linalg.solve(F_new.T,np.linalg.solve(np.linalg.inv(h2), np.array([1,0,0])))
    print('Epipole of img 1', e1)
    print('Epipole of img 2', e2)

    F_mod = F_new
    return F_mod



#=======================================================================================
def rectify_stereo_images(img1, img2, h1, h2):
    # Your code here
    # You should get un-cropped image.
    # In order to superpose two rectified images, you need to create certain amount of margin.
    # Which means you need to do some additional things to get fully warped image (not cropped).
    ################################################
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    cornerPoints = np.array([[0,0],[0, img1.shape[1]], [img1.shape[0], 0], [img1.shape[0], img1.shape[1]]], dtype=float)
    points = np.reshape(cornerPoints, [1,-1,2])

    output_1 = cv2.perspectiveTransform(points, h1)
    output_2 = cv2.perspectiveTransform(points, h2)

    minX = np.min([output_1[:,:,0], output_2[:,:,0]])
    minY = np.min([output_1[:,:,1], output_2[:,:,1]])
    maxX = np.max([output_1[:,:,0], output_2[:,:,0]])
    maxY = np.max([output_1[:,:,1], output_2[:,:,1]])

    ratioX1 = img1.shape[0]/(maxX-minX)
    ratioY1 = img1.shape[1]/(maxY-minY)

    #ratioX1 = 1/ratioX1
    #ratioX2 = 1/ratioX2
    #ratioY1 = 1/ratioY1
    #ratioY2 = 1/ratioY2

    scaling = np.diag([ratioX1, ratioY1, 1])

    scaling[0,2] = -minX
    scaling[1,2] = -minY

    h1 = scaling @ h1
    h2 = scaling @ h2

    h1_mod = h1
    h2_mod = h2
    rectified_size = (img1.shape[1], img1.shape[0])    
    ################################################
    # DO NOT modify lines below!!!
    img1_rectified = cv2.warpPerspective(img1, h1_mod, rectified_size)
    img2_rectified = cv2.warpPerspective(img2, h2_mod, rectified_size)

    return img1_rectified, img2_rectified, h1_mod, h2_mod




#=======================================================================================
def calculate_disparity_map(img1, img2):
    # First convert color image to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    return calculate_disparity_matrix_parallel(img1_gray, img2_gray)
    #return calculate_disparity_matrix_SAD(img1_gray, img2_gray)


#=======================================================================================
# Anything else:

def calculate_disparity_matrix_SAD(img1, img2):
    rows, cols = img1.shape

    disparity_maps = np.zeros([rows, cols, DISPARITY_RANGE])

    for d in range(0, DISPARITY_RANGE):
        # shift image
        translation_matrix = np.float32([[1, 0, d], [0, 1, 0]])
        shifted_image = cv2.warpAffine(img2, translation_matrix,(cols, rows))

        #SAD as an faster alternative to NCC (cause NCC is not running in reasonable time on my computer)
        value = abs(np.float32(img1) - np.float32(shifted_image))
        disparity_maps[:,:,d] = value

    disparity_maps = disparity_maps/DISPARITY_RANGE
    disparity_maps = costAggregation(disparity_maps)
    disparity = np.argmin(disparity_maps, axis=2)

    disparity = -disparity

    return disparity

def worker_new(d, img1, img2, rows, cols, window):
    print(d)
    # shift image
    translation_matrix = np.float32([[1, 0, d], [0, 1, 0]])
    shifted_image = cv2.warpAffine(img2, translation_matrix,(cols, rows))

    img1_windows = np.lib.stride_tricks.sliding_window_view(img1, window)
    img2_windows = np.lib.stride_tricks.sliding_window_view(shifted_image, window)

    result = [[NCC(img1_windows[x,y,:,:], img2_windows[x,y,:,:]) for y in range(WINDOW_SIZE, cols-WINDOW_SIZE)] for x in range(WINDOW_SIZE, rows-WINDOW_SIZE)]
    return result

def calculate_disparity_matrix_parallel(img1_gray, img2_gray):
    rows, cols = img1_gray.shape
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    window = (WINDOW_SIZE, WINDOW_SIZE)
    img1_gray = np.float32(img1_gray)
    img2_gray = np.float32(img2_gray)

    maps = np.array(pool.map(functools.partial(worker_new, img1=img1_gray, img2=img2_gray, rows=rows,cols=cols, window=window), range(DISPARITY_RANGE)))
    disparity_maps = np.dstack(maps)

    disparity_maps = costAggregation_parallel(pool, disparity_maps)
    pool.close()
    pad_with=((WINDOW_SIZE,WINDOW_SIZE),(WINDOW_SIZE,WINDOW_SIZE),(0,0))
    disparity = np.pad(disparity_maps,pad_width=pad_with, constant_values=0)
    disparity = np.argmax(disparity, axis=2)
    disparity = -disparity
    return disparity

def NCC(A, B):
    A = A - np.mean(A)
    B = B - np.mean(B)

    #Avoid dividing by zero
    if(np.sum(A) == 0 or np.sum(B) == 0):
        return 0
    
    result = np.divide(np.sum(A*B), (np.sqrt(np.sum(A*A)) * np.sqrt(np.sum(B*B))))
    return result

def agg_worker(d, volume, window, margin):
    print(d)
    volume_window = np.lib.stride_tricks.sliding_window_view(volume[:,:,d], window)
    volume_window_sum = np.sum(volume_window, axis=(2,3))
    return np.pad(volume_window_sum.astype(np.float32),pad_width=margin, constant_values=0.0)

def costAggregation_parallel(pool, volume):
    window = (AGG_FILTER_SIZE, AGG_FILTER_SIZE)
    margin = int(AGG_FILTER_SIZE/2)
    maps = np.array(pool.map(functools.partial(agg_worker, volume=volume, window=window, margin=margin), range(DISPARITY_RANGE)))
    volume = np.dstack(maps)
    return volume

def costAggregation(volume):
    window = (AGG_FILTER_SIZE, AGG_FILTER_SIZE)
    margin = int(AGG_FILTER_SIZE/2)

    for d in range(DISPARITY_RANGE):
        volume_window = np.lib.stride_tricks.sliding_window_view(volume[:,:,d], window)
        volume_window_sum = np.sum(volume_window, axis=(2,3))
        volume[:,:,d] = np.pad(volume_window_sum,pad_width=margin, constant_values=0)
    return volume
            
