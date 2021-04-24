from typing import Tuple

import cv2
import numpy as np
from skimage.measure import label, regionprops


def imread(name: str) -> np.ndarray:
    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        return img[:, :, [2, 1, 0]] / 255
    else:
        return img[:, :, [2, 1, 0, 3]] / 65535


def imwrite(im: np.ndarray, name: str) -> None:
    im[im < 0] = 0
    im[im > 1] = 1
    cv2.imwrite(name, im[:, :, [2, 1, 0]] * 255)


def fmo_detect_maxarea(
    im: np.ndarray, bgr: np.ndarray, maxarea: float = 0.1
) -> Tuple[np.ndarray, float]:
    # simulate FMO detector -> find approximate location of FMO
    dI = (np.sum(np.abs(im - bgr), 2) > maxarea).astype(float)
    labeled = label(dI)
    regions = regionprops(labeled)
    ind = -1
    maxarea = 0
    for ki in range(len(regions)):
        if regions[ki].area > maxarea:
            ind = ki
            maxarea = regions[ki].area
    if ind == -1:
        return np.array([]), 0
    bbox = np.array(regions[ind].bbox).astype(int)
    return bbox, regions[ind].minor_axis_length


def extend_bbox(
    bbox: np.ndarray,
    ext: float,
    aspect_ratio: float,
    shp: Tuple[int, ...],
) -> np.ndarray:
    height, width = bbox[2] - bbox[0], bbox[3] - bbox[1]

    h2 = height + ext

    h2 = int(np.ceil(np.ceil(h2 / aspect_ratio) * aspect_ratio))
    w2 = int(h2 / aspect_ratio)

    wdiff = w2 - width
    wdiff2 = int(np.round(wdiff / 2))
    hdiff = h2 - height
    hdiff2 = int(np.round(hdiff / 2))

    bbox[0] -= hdiff2
    bbox[2] += hdiff - hdiff2
    bbox[1] -= wdiff2
    bbox[3] += wdiff - wdiff2
    bbox[bbox < 0] = 0
    bbox[2] = np.min([bbox[2], shp[0] - 1])
    bbox[3] = np.min([bbox[3], shp[1] - 1])
    return bbox


def rgba2hs(rgba: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    return rgba[:, :, :3] * rgba[:, :, 3:] + bgr[:, :, :, None] * (
        1 - rgba[:, :, 3:]
    )


def crop_resize(
    Is: np.ndarray, bbox: np.ndarray, res: Tuple[int, int]
) -> np.ndarray:
    rev_axis = False
    if len(Is.shape) == 3:
        rev_axis = True
        Is = Is[:, :, :, np.newaxis]
    imr = np.zeros((res[1], res[0], 3, Is.shape[3]))
    for kk in range(Is.shape[3]):
        im = Is[bbox[0] : bbox[2], bbox[1] : bbox[3], :, kk]
        imr[:, :, :, kk] = cv2.resize(im, res, interpolation=cv2.INTER_CUBIC)
    if rev_axis:
        imr = imr[:, :, :, 0]
    return imr


def rev_crop_resize(
    inp: np.ndarray, bbox: np.ndarray, im: np.ndarray
) -> np.ndarray:
    est_hs = np.tile(im.copy()[:, :, :, np.newaxis], (1, 1, 1, inp.shape[3]))
    for hsk in range(inp.shape[3]):
        est_hs[bbox[0] : bbox[2], bbox[1] : bbox[3], :, hsk] = cv2.resize(
            inp[:, :, :, hsk],
            (bbox[3] - bbox[1], bbox[2] - bbox[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return est_hs
