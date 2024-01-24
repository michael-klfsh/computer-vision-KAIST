import cv2
import numpy as np
from numpy import linalg

from distance import pdist
from feature_extraction import feature_extraction


def get_bags_of_words(image_paths, feature):
    """
    This function assumes that 'vocab_*.npy' exists and contains an vocab size x feature vector
    length matrix 'vocab' where each row is a kmeans centroid or visual word. This
    matrix is saved to disk rather than passed in a parameter to avoid recomputing
    the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size') below.
    """
    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]
    hists = np.zeros((len(image_paths), vocab_size))

    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])
        descriptors = feature_extraction(img, feature)
        distances = pdist(descriptors, vocab)
        nearest_cluster_index = np.argsort(distances, axis=1)[:,0]
        hist = np.zeros((vocab_size))

        for j in range(vocab_size):
            hist[j] = np.count_nonzero(nearest_cluster_index == j)
        l2_norm = np.sqrt(np.sum(np.power(hist, 2)))
        hists[i,:] = hist/l2_norm        

    return hists
