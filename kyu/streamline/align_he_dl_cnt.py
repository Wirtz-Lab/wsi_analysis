import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import numpy as np
from skimage.morphology import closing, square, remove_small_objects, remove_small_holes
from math import atan2, degrees
import os
import cv2
from openslide import OpenSlide
from rotate_image import rotate_image
from matplotlib import pyplot as plt

def align_he_dl_cnt(dst,fn,wsisrc,dlsrc,cntsrc,roisrc):
    roi = Image.open(os.path.join(roisrc, '{}.{}'.format(fn, 'png')))  # roi is very small
    roiarr = np.array(roi)
    numsecmax = np.max(roiarr)
    imcropdst = os.path.join(dst, 'imcrop')
    dlcropdst = os.path.join(dst, 'dlcrop')
    nuccropdst = os.path.join(dst, 'nuccrop')
    if not os.path.exists(dst): os.mkdir(dst)
    if not os.path.exists(imcropdst): os.mkdir(imcropdst)
    if not os.path.exists(dlcropdst): os.mkdir(dlcropdst)
    if not os.path.exists(nuccropdst): os.mkdir(nuccropdst)

    dstfn = fn + 'sec{}'.format(numsecmax) + '.png'
    if os.path.exists(os.path.join(imcropdst, dstfn)):return

    mask = Image.open(os.path.join(dlsrc, '{}.{}'.format(fn, 'tif')))
    ndpi = OpenSlide(os.path.join(wsisrc, '{}.{}'.format(fn, 'ndpi')))
    json = pd.read_json(os.path.join(cntsrc, '{}.{}'.format(fn, 'json')), orient='index')

    [x, y] = roi.size
    (w, h) = ndpi.level_dimensions[0]
    rsf = [w / x, h / y]
    rsf = rsf[0]

    json = pd.DataFrame(json[0].loc['nuc']).T.drop(columns=['type_prob'])
    json = json[json['contour'].map(len) > 5].reset_index(drop=True)

    def isinroi(row):
        newrow = [round(_ / 16) for _ in row]
        return roiarr[newrow[1], newrow[0]]
    json['inroi'] = json['centroid'].apply(lambda row: isinroi(row))
    jsoninroi = json[json['inroi'] > 0]

    # create binary labeled mask of nuclei within roi
    nuc_image = np.zeros((h, w), dtype=np.int32)  # need to flip h and w
    for idx, ct in enumerate(jsoninroi['contour']):
        cv2.fillPoly(nuc_image, pts=[np.array(ct).astype(np.int32)], color=idx + 1)

    # create DLmask within roi
    mask_resized = mask.resize(roi.size, resample=0)  # nearest interpolation to rescale
    DLsmall = np.array(mask_resized)
    DLinroi = np.multiply(DLsmall, roiarr > 0)



    minDermhole = 5000
    minepisize = 1000
    whitespace = 12

    epi = (DLinroi == 1) | (DLinroi == 2)
    derm = (2 < DLinroi) & (DLinroi < whitespace)
    derm = remove_small_holes(derm, area_threshold=minDermhole)

    epi2 = epi & ~derm
    epi2 = remove_small_objects(epi2, min_size=minepisize, connectivity=2)


    fns =[]
    secNs =[]
    drots =[]
    bboxs=[]
    bbox2s=[]

    for numsec in range(1, numsecmax+1):
        print('section N: ', numsec, '/', numsecmax)
        roitmp = roiarr == numsec  # roitmp is logical, not greyscale
        mskepi = roitmp & epi2
        DLtmp = np.multiply(DLinroi, roitmp)  # roiscale

        # create H&E
        [xroi, yroi] = np.where(roitmp)
        bboxroi = [np.min(xroi), np.max(xroi), np.min(yroi), np.max(yroi)]
        roitmp2 = roitmp[bboxroi[0]:bboxroi[1], bboxroi[2]:bboxroi[3]]

        bboxroi = [round(_ * rsf) for _ in bboxroi]
        # targetlevel = ndpi.get_best_level_for_downsample(rsf)
        HE = ndpi.read_region(location=(bboxroi[2], bboxroi[0]), level=0, size=(bboxroi[3]-bboxroi[2],bboxroi[1]-bboxroi[0]))
        roitmp2 = cv2.resize(roitmp2.astype(np.uint8), dsize=HE.size, interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((5, 5), np.uint8)
        roitmp2 = cv2.dilate(roitmp2, kernel, iterations=1)
        HEtmp = np.multiply(np.array(HE)[:, :, :3], np.repeat(roitmp2[:, :, np.newaxis], 3, axis=2))  # roiscale

        # align horizontal
        [xt0, yt0] = np.where(mskepi)
        vertices = np.array([xt0[::10], yt0[::10]]).T
        vc = vertices - vertices.mean(axis=0)

        U, S, Vt = np.linalg.svd(vc)
        k = Vt.T
        d0 = degrees(atan2(k[1, 1], k[1, 0]))
        if np.linalg.det(k) < 0: d0 = -d0
        if d0 < 0: d0 = d0 + 360
        # clear variables, we just need d0
        del vertices, vc, U, S, Vt, k, mskepi

        # enlarge DLtmp from roiscale to 20x wsi scale
        DLtmplarge = cv2.resize(DLtmp.astype(np.uint8), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        [xt, yt] = np.where(DLtmplarge[:, :])
        bbox = [np.min(xt), np.max(xt), np.min(yt), np.max(yt)]
        bbox = [round(_) for _ in bbox]
        DLrot = DLtmplarge[bbox[0]:bbox[1], bbox[2]:bbox[3]]  # crop to save memory for practical rotation

        try:DLrot = rotate_image(DLrot, d0)  # size of an input and output images should be less than 32767x32767
        except:return [DLrot,d0]

        # check if section is upside-down
        [xderm, yderm] = np.where(DLrot)
        [xepi, yepi] = np.where((DLrot == 1) | (DLrot == 2))
        if np.mean(xderm) - np.mean(xepi) < 0:  # if dermis is above epidermis, then flip
            DLrot = np.rot90(np.rot90(DLrot))
            d0 += 180

        # crop again to remove border effect
        [xt2, yt2] = np.where(DLrot)
        bbox2 = [np.min(xt2), np.max(xt2), np.min(yt2), np.max(yt2)]
        bbox2 = [round(_) for _ in bbox2]
        DLrot = DLrot[bbox2[0]:bbox2[1], bbox2[2]:bbox2[3]]

        # crop nuclei labeled binary image using roi/DLtmplarge
        nuctmp = nuc_image[bbox[0]:bbox[1], bbox[2]:bbox[3]]  # crop
        nucrot = rotate_image(nuctmp, d0)  # rotate
        nucrot = nucrot[bbox2[0]:bbox2[1], bbox2[2]:bbox2[3]]  # crop

        # crop HE
        # bboxsmall = [round(_ / rsf) for _ in bbox]
        # bboxsmall2 = [round(_ / rsf) for _ in bbox2]
        HEtmp = HEtmp[bbox[0]:bbox[1], bbox[2]:bbox[3]]  # crop
        HErot = rotate_image(HEtmp, d0)  # rotate
        HErot = HErot[bbox2[0]:bbox2[1], bbox2[2]:bbox2[3]]  # crop
        HErot[HErot == 0] = 235
        # HEbig = cv2.resize(HErot.astype(np.uint8), dsize=[_ for _ in DLrot.shape][::-1],
        #                    interpolation=cv2.INTER_NEAREST)
        if not HErot.shape[:-1] == DLrot.shape:
            raise "need to check cropping"

        dstfn = fn + 'sec{}'.format(numsec) + '.png'

        # save images
        Image.fromarray(HErot).save(
            os.path.join(imcropdst, dstfn))
        Image.fromarray(DLrot.astype('uint32')).save(
            os.path.join(dlcropdst, dstfn))
        Image.fromarray(nucrot).save(
            os.path.join(nuccropdst, dstfn))
        fns.append(fn)
        secNs.append(numsec)
        drots.append(d0)
        bboxs.append(bbox)
        bbox2s.append(bbox2)
    rotationLUT = pd.DataFrame({'fn':fns,'secN':secNs,'drot':drots,'bbox':bboxs,'bbox2':bbox2s})
    return rotationLUT








