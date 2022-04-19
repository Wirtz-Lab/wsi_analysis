from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import numpy as np
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects, remove_small_holes, binary_dilation
from skimage.transform import rotate
from math import atan2, degrees
import os
from copy import deepcopy
import cv2
from time import time

def crop_align_DL(imname):
    minTA = 10000
    minTAhole = 100
    minDermhole = 5000
    minepisize=1000
    whitespace=12
    src = os.path.dirname(imname)
    dst = os.path.join(src, 'crop_TA')
    if not os.path.exists(dst): os.mkdir(dst)
    fn, ext = os.path.splitext(os.path.basename(imname))
    if os.path.exists(os.path.join(dst, '{}_sec{:02d}.png'.format(fn, 4))):
        print('continue')
        return

    # open image
    im = Image.open(imname)
    TAbig = np.array(im)
    # downsize to expedite
    (width, height) = (im.width // 10, im.height // 10)
    im_resized = im.resize((width, height), resample=0)
    TA = np.array(im_resized)
    sure_fg = closing((2 < TA) & (TA < whitespace - 1), square(3))  # 13sec
    sure_fg = remove_small_objects(sure_fg, min_size=minTA, connectivity=2)  # 6sec
    sure_fg = remove_small_holes(sure_fg, area_threshold=minTAhole / 100).astype(np.uint8)  # 7sec
    # define background
    bw = closing(TA < whitespace, square(3))  # 12 is background
    bw = remove_small_objects(bw, min_size=minTA, connectivity=2)
    bw = remove_small_holes(bw, area_threshold=minTAhole)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bw.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)  # 2sec
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # define middleground
    unknown = cv2.subtract(sure_bg, sure_fg).astype(np.bool)
    # label that background is 1 and objects are 2~N and middleground is zero
    sure_fg_label = label(sure_fg).astype(np.int32)
    sure_fg_label = sure_fg_label + 1
    sure_fg_label[unknown] = 0
    # perform watershed based on the marker
    TAbgr = cv2.cvtColor(TA, cv2.COLOR_GRAY2BGR)
    label_image = cv2.watershed(TAbgr, sure_fg_label)
    # iterate each section
    epi = (TA == 1) | (TA == 2)
    derm = (2 < TA) & (TA < whitespace)
    derm = remove_small_holes(derm, area_threshold=minDermhole)
    epi2 = epi & ~derm
    epi2 = remove_small_objects(epi2, min_size=minepisize, connectivity=2)
    numsecmax = np.max(label_image)
    for numsec in range(1,numsecmax):
        print('section N: ', numsec, '/', numsecmax-1)
        msktmp = label_image == numsec+1
        # mskderm = msktmp & derm
        mskepi = msktmp & epi2
        # align horizontal
        [xt2, yt2] = np.where(mskepi)
        vertices = np.array([xt2[::10], yt2[::10]]).T
        vc = vertices - vertices.mean(axis=0)
        U, S, Vt = np.linalg.svd(vc)
        k = Vt.T
        d0 = -degrees(atan2(k[0, 1], k[0, 0]))
        TAtmp = deepcopy(TAbig)
        mskbig = cv2.resize(msktmp.astype(np.uint8), TAtmp.shape[::-1], interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((20, 20), np.uint8)
        mskbig = cv2.dilate(mskbig, kernel, iterations=3)
        TAtmp[mskbig == 0] = 0 #scale back up to perform rotation #1sec

        start = time() # 10sec
        degrot = np.abs(d0 - 90) # TO-DO: confirm if this is true.
        mskrot = rotate(TAtmp, degrot, resize=True, preserve_range=True, order=0)  # this is slow
        #can I expedite by not preserving range and recovering original pixel later?
        print(round(time() - start), 'sec elapsed for part A')

        start = time() #
        [xt, yt] = np.where(mskrot) # mskrot is sometimes not detected
        [xt2, yt2] = np.where((mskrot == 1) | (mskrot == 2))
        mskrot2 = mskrot[np.min(xt):np.max(xt), np.min(yt):np.max(yt)]
        print(round(time() - start), 'sec elapsed for part B')
        start = time()  #
        if np.mean(xt) - np.mean(xt2) < 0:  # if dermis is above epidermis, flip it
            mskrot2 = np.rot90(np.rot90(mskrot2))
        print(round(time() - start), 'sec elapsed for part C')
        start = time()  #
        mskrot2[mskrot2 == 0] = whitespace  # assign whitespace value to background
        print(round(time() - start), 'sec elapsed for part D')

        Image.fromarray(mskrot2.astype('int8')).save(
            os.path.join(dst, '{}_sec{:02d}.png'.format(fn, numsec)))


