import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_*.npy' exists and
    contains an vocab size x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """

    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    hists = np.zeros((len(image_paths), int((1/3)*(np.power(4, max_level+1)-1) * vocab_size)))
    print(hists.shape)
    for image_id in range(len(image_paths)):
        img = cv2.imread(image_paths[image_id])
        hist = np.empty((0))
        for i in range(max_level+1):
            side = int(np.sqrt(4**i))
            M = img.shape[0] // side
            N = img.shape[1] // side

            w=1
            if(i == 0):
                w = w**(-max_level)
            else:
                w = w**(-max_level+i-1)

            tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]

            for j in range(0, 4**i):
                patch_hist = np.zeros((vocab_size))
                patch = tiles[j]
                descriptors = feature_extraction(patch, feature)
                if(descriptors.shape[0] != 0):
                    distances = pdist(descriptors, vocab)
                    nearest_cluster_index = np.argsort(distances, axis=1)[:,0]
                else:
                    nearest_cluster_index = np.zeros((1))

                for cat_idx in range(vocab_size):
                    patch_hist[cat_idx] = np.count_nonzero(nearest_cluster_index == cat_idx)
                patch_hist = w * patch_hist
                hist = np.concatenate((hist, patch_hist))
        hists[image_id,:] = np.array(hist)    

    return hists
