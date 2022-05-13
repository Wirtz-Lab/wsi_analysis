
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from skimage.morphology import remove_small_objects
import numpy as np
import cv2

# dl = r'\\fatherserverdw\Q\research\images\skin_aging\1um\classification_v9_combined\12.tif'
# roi = r'\\fatherserverdw\Q\research\images\skin_aging\annotation\roi\tif\12_tissue_binary.tif'
#
# dl = Image.open(dl)
# roi = Image.open(roi)

def dl2distancemap(roi,dl):
    roi = roi.resize(dl.size)
    roiarr = np.array(roi)
    dlarr = np.array(dl)
    dlarr = dlarr*roiarr
    minszs = [100000,100000,1000,1000,1000,1000,1000,1000,1000,10000,1000] # 11 elements
    dist_layers = np.zeros_like(dlarr).astype(np.float16)
    dist_layers = np.repeat(dist_layers[np.newaxis,...], len(minszs), axis=0)
    for idx,minsz in enumerate(minszs):
        dltmp = dlarr==idx+1
        dltmp2 = remove_small_objects(dltmp,minsz)
        dist = cv2.distanceTransform(np.invert(dltmp2).astype(np.uint8), cv2.DIST_L2, 3)
        dist_layers[idx,...] = dist
    return dist_layers