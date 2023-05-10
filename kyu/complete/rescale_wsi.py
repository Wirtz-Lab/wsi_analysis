from openslide import OpenSlide
from PIL import Image
import numpy as np
import os
from time import time
from natsort import natsorted
import pandas as pd

def svs2tiff(svs,rsf):
    src,fn = os.path.split(svs)
    fn,ext = os.path.splitext(fn)
    fn1 = fn + '.tif'
    if os.path.exists(os.path.join(svs_dst,fn1)): return
    print(fn)
    svs_obj = OpenSlide(svs)
    svs_img = svs_obj.read_region(location=(0,0),level=0,size=svs_obj.level_dimensions[0]).convert('RGB')
    resize_factorx = rsf/float(svs_obj.properties['openslide.mpp-x']) #8um = 1.25x #4um = 2.5x, #2um=5x, 1um=10x, 0.5um=20x, 0.25um=40x
    resize_factory = rsf/float(svs_obj.properties['openslide.mpp-y'])
    resize_dimension = tuple([int(np.ceil(svs_obj.dimensions[0]/resize_factorx)),int(np.ceil(svs_obj.dimensions[1]/resize_factory))])
    svs_img = svs_img.resize(resize_dimension,resample=Image.NEAREST)
    svs_img.save(os.path.join(svs_dst,fn1),resolution=1,resolution_unit=1,quality=100,compression=None)

svs_src = r'\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HS-012-D9\raw images'
svs_dst = r'\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HS-012-D9\1um'
imlist = [_ for _ in os.listdir(svs_src) if _.endswith('ndpi')]
imlist = natsorted(imlist)
#
# xl = pd.read_excel(r"\\fatherserverdw\kyuex\imlist_all.xlsx",usecols=['filename','body part 1','student score'])
# backxl = xl[xl['body part 1'].str.lower()=='back']
# healthybackxl = backxl[backxl['student score']>1]

if not os.path.exists(svs_dst):
    os.mkdir(svs_dst)

st = time()

for idx,svs in enumerate(imlist):
    svs2tiff(os.path.join(svs_src,svs),1)
    print(idx,'/',len(imlist))

print("{:.2f} sec elapsed for {:d} images at 10x".format(time()-st,len(imlist)))