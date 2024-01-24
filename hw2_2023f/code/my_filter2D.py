from ctypes import FormatError
from os import error
import cv2
import numpy as np
from skimage.util.shape import view_as_windows

def reflected_pad(image, lr_extend, ud_extend):
    pad_left = np.fliplr(image[:,1:lr_extend+1,:])
    pad_right = np.fliplr(image[:,-lr_extend-1:-1,:])
    image = np.concatenate((pad_left, image, pad_right), axis=1)
    pad_top = np.flipud(image[1:ud_extend+1,:,:])
    pad_down = np.flipud(image[-ud_extend-1:-1,:,:])
    image = np.concatenate((pad_top, image, pad_down), axis=0)
    return image


def my_filter2D(image, kernel, bordertype='ZEROS'):
    # This function computes convolution given an image and kernel.
    # While "correlation" and "convolution" are both called filtering, here is a difference;
    # 2-D correlation is related to 2-D convolution by a 180 degree rotation of the filter matrix.
    #
    # Your function should meet the requirements laid out on the project webpage.
    #
    # Boundary handling can be tricky as the filter can't be centered on pixels at the image boundary without parts
    # of the filter being out of bounds. If we look at BorderTypes enumeration defined in cv2, we see that there are
    # several options to deal with boundaries such as cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, etc.:
    # https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    #
    # Your my_filter2D() computes convolution with the following behaviors:
    # - to pad the input image with zeros,
    # - and return a filtered image which matches the input image resolution.
    # - A better approach is to mirror or reflect the image content in the padding (borderType=cv2.BORDER_REFLECT_101).
    #
    # You may refer cv2.filter2D() as an exemplar behavior except that it computes "correlation" instead.
    # https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)   # for extra credit
    # Your final implementation should not contain cv2.filter2D().
    # Keep in mind that my_filter2D() is supposed to compute "convolution", not "correlation".
    #
    # Feel free to add your own parameters to this function to get extra credits written in the webpage:
    # - pad with reflected image content
    # - FFT-based convolution

    #Error case
    if(kernel.shape[0]%2 == 0 or kernel.shape[1]%2 == 0):
        return FormatError("No even kernel shape!")
    # used to store the convoluted image
    h = np.zeros(image.shape)
    
    lr_extend = int(kernel.shape[1]/2)
    ud_extend = int(kernel.shape[0]/2)

    if(bordertype == "ZEROS"):
        #Add padding of zeros
        image = np.pad(image, ((ud_extend, ud_extend),(lr_extend, lr_extend), (0,0)), mode='constant', constant_values=0)
    elif(bordertype == "REFLECT"):
        image = reflected_pad(image, lr_extend, ud_extend)

    for layer in range(image.shape[2]):
        image_adj = image[:,:,layer].reshape((image.shape[0], image.shape[1]))
        #Flip the kernel along both axis to perform a convolution
        kernel_fliped = np.flipud(np.fliplr(kernel))

        #Create all subimages of kernel size
        subimages = view_as_windows(image_adj, kernel_fliped.shape)
        for i in range(subimages.shape[0]):
            for j in range(subimages.shape[1]):
                #Apply filter and sum up
                result = np.multiply(subimages[i][j],kernel_fliped).sum()
                h[i][j][layer] = result
    return h
