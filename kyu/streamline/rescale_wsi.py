from openslide import OpenSlide
from PIL import Image
import numpy as np
import os

def rescale_wsi(svs,rsf):
    src,fn = os.path.split(svs)
    fn,ext = os.path.splitext(fn)
    wsi_dst = os.path.join(src,'{}um'.format(rsf))
    if not os.path.exists(wsi_dst): os.mkdir(wsi_dst)
    fn1 = fn + '.tif'
    if os.path.exists(os.path.join(wsi_dst,fn1)): return
    print(fn)
    svs_obj = OpenSlide(svs)
    svs_img = svs_obj.read_region(location=(0,0),level=0,size=svs_obj.level_dimensions[0]).convert('RGB')
    resize_factorx = rsf/float(svs_obj.properties['openslide.mpp-x']) #8um = 1.25x #4um = 2.5x, #2um=5x, 1um=10x, 0.5um=20x, 0.25um=40x
    resize_factory = rsf/float(svs_obj.properties['openslide.mpp-y'])
    resize_dimension = tuple([int(np.ceil(svs_obj.dimensions[0]/resize_factorx)),int(np.ceil(svs_obj.dimensions[1]/resize_factory))])
    svs_img = svs_img.resize(resize_dimension,resample=Image.NEAREST)
    svs_img.save(os.path.join(wsi_dst,fn1),resolution=1,resolution_unit=1,quality=100,compression=None)