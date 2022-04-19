from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import os
from openslide import OpenSlide
import numpy as np
from skimage.morphology import closing, square, remove_small_objects, remove_small_holes
from skimage.segmentation import clear_border

def wsi2TA(impth):
    rsf = 128
    src = os.path.dirname(impth)
    dst = os.path.join(src,'TA')
    if not os.path.exists(dst): os.mkdir(dst)
    imnm,ext = os.path.splitext(os.path.basename(impth))
    im = OpenSlide(impth)
    target = im.get_best_level_for_downsample(rsf)
    imdim = im.level_dimensions[target]
    imsmall = im.read_region(location=(0, 0), level=target, size=imdim)
    bw = (150 < np.array(imsmall)[:, :, 0]) & (np.array(imsmall)[:, :, 1] < 210)
    bw = closing(bw, square(3))
    minTA = 8000
    bw = remove_small_objects(bw, min_size=minTA, connectivity=2)
    minTAhole = 4000
    bw = remove_small_holes(bw, area_threshold=minTAhole)
    bw = clear_border(bw)
    fn, ext = os.path.splitext(imnm)
    Image.fromarray(bw).save(os.path.join(dst, '{}.png'.format(fn)))
    return bw