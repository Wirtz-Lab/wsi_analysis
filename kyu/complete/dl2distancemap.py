from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from skimage.morphology import remove_small_objects
import numpy as np
import cv2

# output precision is set to float16 manually here since our image is smaller than 16bit max: 65504.
# consider using 32bit if we use 40x images that can be bigger than 65504 in one dimension.
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
        dist = cv2.distanceTransform(np.invert(dltmp2).astype(np.uint8), cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        #DO NOT use masksize 3 for L2. It is NOT accurate at all.
        #use maskSize of 5 or cv2.DIST_MASK_PRECISE. Note: precise is faster than 5.
        dist_layers[idx,...] = dist
    return dist_layers