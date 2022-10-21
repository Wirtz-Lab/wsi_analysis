from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from skimage.morphology import remove_small_objects
import numpy as np
import cv2
from skimage.measure import label

# output precision is set to float16 manually here since our image is smaller than 16bit max: 65504.
# consider using 32bit if we use 40x images that can be bigger than 65504 in one dimension.

# must scale down input image to avoid crazy processing time
def dl2distancemap_byroiv2(roi,dl):

    roiimL = np.array(roi)
    # roiimL = label(roiarr)
    numsec = np.max(roiimL)

    #we are gonna resize dl to roi size for faster calculations
    dl = dl.resize(roi.size)
    dlarr = np.array(dl)

    minszs = [8000,100000,1000,1000,1000,1000,1000,1000,1000,10000,1000] # 11 elements
    minszs = [round(_ / 16) for _ in minszs] #downscale by 16 from 0.454um/px to 2um/px

    # dist_layers = np.repeat(dist_layers[np.newaxis, ...], numsec, axis=0) #pre-allocate stacks for roi #oom issue
    # print(dist_layers.shape) #sanity check for output dimension

    #iterate each roi label
    dist_layers2=[]
    for roidx in range(1,numsec+1):
        dlarrm = np.multiply(dlarr, roiimL == roidx)
        dist_layers = np.zeros_like(dlarrm).astype(np.float16)  # pre-allocate x,y single layer
        dist_layers = np.repeat(dist_layers[np.newaxis, ...], len(minszs),axis=0)  # pre-allocate stacks for tissue classes
        for idx,minsz in enumerate(minszs):
            dltmp = dlarrm==idx+1
            dltmp2 = remove_small_objects(dltmp,minsz)
            dist = cv2.distanceTransform(np.invert(dltmp2).astype(np.uint8), cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
            #DO NOT use masksize 3 for L2. It is NOT accurate at all.
            #use maskSize of 5 or cv2.DIST_MASK_PRECISE. Note: precise is faster than 5.
            dist_layers[idx,...] = dist
        dist_layers2.append(dist_layers)
    return dist_layers2