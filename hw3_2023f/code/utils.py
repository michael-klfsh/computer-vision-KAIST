import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import colormaps as mcolormaps
from matplotlib import colors as mcolors

def homogenize(x, axis = -1):
    """
    Parameters:
        x: np.ndarray
        axis: int
    Returns:
        result: np.ndarray,
                `x.shape[i] == result.shape[i]` if `i != axis`
                `x.shape[i] + 1 == result.shape[i]` if `i == axis`
    -----
    Examples
    ```
    >>> homogenize(np.array([2, 3]))
    array([2, 3, 1])
    >>> homogenize(np.array([[2, 3], [4, 5], [6, 7]]))
    array([[2, 3, 1],
          [4, 5, 1],
          [6, 7, 1]])
    >>> homogenize(np.array([[1, 2], [3, 4]]), axis=0)
    array([[1, 2],
          [3, 4],
          [1, 1]])
    ```
    """
    sh = list(x.shape)
    sh[axis] = 1
    one = np.ones_like(x, shape=sh) # inherit `x.dtype`
    result = np.concatenate([x, one], axis=axis)
    return result


def normalize_points(pts, numDims): 
    # strip off the homogeneous coordinate
    points = pts[:numDims,:]

    # compute centroid
    cent = np.mean(points, axis=1)

    # translate points so that the centroid is at [0,0]
    translatedPoints = np.transpose(points.T - cent)

    # compute the scale to make mean distance from centroid sqrt(2)
    meanDistanceFromCenter = np.mean(np.sqrt(np.sum(np.power(translatedPoints,2), axis=0)))
    if meanDistanceFromCenter > 0: # protect against division by 0
        scale = np.sqrt(numDims) / meanDistanceFromCenter
    else:
        scale = 1.0

    # compute the matrix to scale and translate the points
    # the matrix is of the size numDims+1-by-numDims+1 of the form
    # [scale   0     ... -scale*center(1)]
    # [  0   scale   ... -scale*center(2)]
    #           ...
    # [  0     0     ...       1         ]    
    # T = np.diag(np.array([*np.ones(numDims) * scale, 1], dtype=np.float))
    T = np.diag(np.array([*np.ones(numDims) * scale, 1], dtype=float))
    T[0:-1, -1] = -scale * cent

    if pts.shape[0] > numDims:
        normPoints = T @ pts
    else:
        normPoints = translatedPoints * scale

    # the following must be true:
    # np.mean(np.sqrt(np.sum(np.power(normPoints[0:2,:],2), axis=0))) == np.sqrt(2)

    return normPoints, T


def compute_epe(disparity, gt_disparity):
    # compute end point error between disparity map and gt
    _epe = np.sqrt((disparity - gt_disparity) ** 2)
    #print(disparity[250,:])
    #print(gt_disparity[250,:])
    epe = _epe.mean()
    epe3 = (_epe > 3.0).astype(np.float32).mean()
    
    return epe, epe3



def evaluate_criteria(extime, epe, epe3):
    if extime < 75.0:
        s = 'PASS'
        
        if epe < 3.0: score_epe = 15
        elif (epe >= 3.0) and (epe < 4.0): score_epe = 10
        elif (epe >= 4.0) and (epe < 5.0): score_epe = 5
        else: score_epe = 0

        if epe3 < 0.25: score_bad_pix = 15
        elif (epe3 >= 0.25) and (epe3 < 0.3): score_bad_pix = 10
        elif (epe3 >= 0.3) and (epe3 < 0.4): score_bad_pix = 5
        else: score_bad_pix = 0
    else:
        s = 'FAIL'
        score_epe = 0
        score_bad_pix = 0
    return s, score_epe, score_bad_pix



##################################################
# Visualization
##################################################

def image_overlay_line(img, line, width=1.0, color='r'):
    """
    Parameters:
        img: np.ndarray[h, w] or [h, w, 3]
        line: np.ndarray[3], which indicates the line of the equation `line[0]*x + line[1]*y + line[2] == 0`.
        width: float, line width
        color: str | tuple | np.ndarray, line color
    Returns:
        result: np.ndarray[h, w] or [h, w, 3], same shape as `img`
    """
    # Preprocessing `line`
    line = np.array(line, dtype=np.float64) # copy
    # line = line.squeeze()
    assert line.shape == (3,)
    line /= np.linalg.norm(line[:2])

    assert img.ndim in [2, 3]
    img = np.array(img)
    h, w = img.shape[:2]

    assert width > 0.0
    color = np.array(mcolors.to_rgb(color))
    if img.dtype == np.uint8:
        color *= 255

    Y, X = np.mgrid[:h, :w]
    # NOTE https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    dist_to_line = np.abs( line[0]*X + line[1]*Y + line[2] )
    alpha = np.where(dist_to_line <= width, 1.0, np.maximum(width - dist_to_line + 1.0, 0.0))
    alpha = alpha[:,:,None]
    # now 0.0 <= alpha <= 1.0

    if img.ndim == 2:
        img = np.broadcast_to(img[:,:,None], (h, w, 3))
    result = (1-alpha) * img + alpha * color
    return result.astype(img.dtype)

def image_overlay_circle(img, center, radius=5.0, color='r'):
    """
    Parameters:
        img: np.ndarray[h, w] or [h, w, 3]
        center: np.ndarray[2] or [3], the center of the circle. [3] as homogeneous coordinates
        radius: float, line width
        color: str | tuple | np.ndarray, line color
    Returns:
        result: np.ndarray[h, w] or [h, w, 3], same shape as `img`

    NOTE `center` has x-y order
    """
    # Preprocessing `center`
    center = np.asarray(center)
    if center.shape == (3,):
        center = center[:2] / center[2]
    elif center.shape != (2,):
        raise ValueError(f"The shape of `center` must be (2,) or (3,), but {center.shape} was given.")

    assert img.ndim in [2, 3]
    img = np.array(img)
    h, w = img.shape[:2]

    # Preprocessing `radius` and `color`
    assert radius > 0.0
    if isinstance(color, str):
        color = mcolors.to_rgb(color)
    color = np.array(color)
    if img.dtype == np.uint8:
        color *= 255

    Y, X = np.mgrid[:h, :w] # [2, h, w]
    XY = np.stack([X, Y], axis=-1)
    dist_to_circle = np.linalg.norm(XY - center, axis=-1)
    alpha = np.where(dist_to_circle <= radius, 1.0, np.maximum(radius - dist_to_circle + 1.0, 0.0))
    alpha = alpha[:,:,None]
    # now 0.0 <= alpha <= 1.0

    if img.ndim == 2:
        img = np.broadcast_to(img[:,:,None], (h, w, 3))
    result = (1-alpha) * img + alpha * color
    return result.astype(img.dtype)

def images_overlay_fundamental(img1, img2, pts1, pts2, F):
    """
    Parameters:
        img1: np.ndarray[h, w, 3], the first image
        img2: np.ndarray[h, w, 3], the second image
        pts1: np.ndarray[N, 2], some point locations in the first image
              (xy pixel coordinates, i.e. its color is `img1[pts1[1], pts1[0], :]`)
        pts2: np.ndarray[N, 2], some point locations in the second image
              (xy pixel coordinates, i.e. its color is `img2[pts2[1], pts2[0], :]`)
        F: np.ndarray[3, 3], fundamental matrix of the two images
           satisfying `x2.T @ F @ x1` is approximately `0`.
    Returns:
        (res1, res2): res1 is the overlayed image from `img1` with points `pts1` and lines obtained by `pts2` and `F`.
                      res2 vice versa 
    """
    # Preprocess
    assert img1.ndim == 3
    assert img2.ndim == 3
    assert pts1.ndim == 2
    assert pts2.ndim == 2
    assert pts1.shape[1] == 2
    assert pts2.shape[1] == 2
    assert pts1.shape[0] == pts2.shape[0]
    assert F.shape == (3, 3)
    res1 = np.array(img1) # copy
    res2 = np.array(img2) # copy
    pts1_h = homogenize(pts1)
    pts2_h = homogenize(pts2)

    # Visualization options
    linewidth = 1.0
    radius = 7.0
    def color(i):
        return mcolormaps['Set1'](i)[:3]
    
    for i, (pt1, pt2, pt1_h, pt2_h) in enumerate(zip(pts1, pts2, pts1_h, pts2_h)):
        """
        NOTE If shapes of `mat1`, `mat2`, and `mat3` are `(M,)`, `(M, N)`, and `(N,)`, resp., then
            `(mat1 @ mat2).shape == (N,)` and `(mat1 @ mat2)[i] == (mat1 * mat2[:, i]).sum()`
            `(mat1 @ mat2).shape == (M,)` and `(mat2 @ mat3)[i] == (mat2[i, :] * mat3).sum()`
        """
        current_color = color(i)
        #print(current_color)
        res1 = image_overlay_line(res1, pt2_h @ F, width=linewidth, color=(current_color))
        res2 = image_overlay_line(res2, F @ pt1_h, width=linewidth, color=current_color)
        res1 = image_overlay_circle(res1, pt1, radius=radius, color=current_color)
        res2 = image_overlay_circle(res2, pt2, radius=radius, color=current_color)
    
    return res1, res2

def get_anaglyph(img1, img2):
    assert img1.shape[2] == 3
    assert img2.shape[2] == 3
    img1_ = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_ = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return np.dstack((img1_, img2_, img2_))

def draw_epipolar_overlayed_img(img1, img2, pts1, pts2, fundamental_matrix):
    img1, img2 = images_overlay_fundamental(img1, img2,
                                            pts1, pts2, fundamental_matrix)
    fig, ax = plt.subplots(1,2, figsize=(20,8))
    ax[0].imshow(img1)
    ax[0].set_title('Left image')
    ax[1].imshow(img2)
    ax[1].set_title('Right image')
    fig.tight_layout()
    return fig


def draw_stereo_rectified_img(img1_rectified, img2_rectified):
    fig, ax = plt.subplots(1,2, figsize=(20,8))
    ax[0].imshow(img1_rectified)
    ax[0].set_title('Left image rectified')
    ax[1].imshow(img2_rectified)
    ax[1].set_title('Right image rectified')
    fig.tight_layout()
    return fig



def draw_disparity_map(disparity_map):
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    img = ax.imshow(disparity_map, cmap='turbo')
    ax.set_title('Disparity map')
    fig.colorbar(img, fraction=0.027, pad=0.01)
    fig.tight_layout()
    return fig