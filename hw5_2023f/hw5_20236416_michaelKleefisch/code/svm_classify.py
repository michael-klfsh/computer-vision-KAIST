import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats:
        an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels:
        an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats:
        an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type:
        the name of a kernel type. 'linear' or 'RBF'.

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)
    C = 1.0
    results = []
    if(kernel_type == "RBF"):
        kernel_type = "rbf"

    svm_instance = svm.SVC(C=C, kernel=kernel_type, decision_function_shape="ovr")
    svm_instance.fit(train_image_feats, train_labels)

    for i in range(test_image_feats.shape[0]):
        img_feature = test_image_feats[i,:]
        img_feature = img_feature.reshape((1,img_feature.shape[0]))

        decision = svm_instance.decision_function(img_feature)[0]
        pred_category = categories[np.argmax(decision)]

        results.append(pred_category)


    return np.array(results)