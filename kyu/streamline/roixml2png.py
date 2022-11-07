import xml.etree.ElementTree as ET
from time import time
import numpy as np
import cv2
import os
import pandas as pd
from openslide import OpenSlide
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from skimage.morphology import closing, square, remove_small_objects, remove_small_holes
from time import time
def roixml2png(xml_path,imsrc):
    # try:
    start = time()
    print(os.path.basename(xml_path))

    fol,fn = os.path.split(xml_path)
    imfn = fn.replace('xml','ndpi')
    mskdst = os.path.join(fol,'labeledmask_20rsf')
    if not os.path.exists(mskdst):
        os.mkdir(mskdst)

    dstfn = os.path.join(mskdst, '{}.png'.format(imfn.replace('.ndpi','')))
    if os.path.exists(dstfn):
        return

    TAdst = os.path.join(fol,'TA_20rsf')
    if not os.path.exists(TAdst):
        os.mkdir(TAdst)
    TAdstfn = os.path.join(TAdst, '{}.png'.format(imfn.replace('.ndpi','')))


    # Open XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Read Class names and put into a list called classlut
    # classlut = []
    # for Annotation in root.iter('Annotation'):
    #     for Attrib in Annotation.iter('Attribute'):
    #         classlut.append(Attrib.attrib.get('Name'))
    # classluts = sorted(classlut)
    classluts = ['tissue']
    dfs = []
    for idx, Annotation in enumerate(root.iter('Annotation')): #iterate each class
        for Region in Annotation.iter('Region'): #iterate each circle
            x = np.array([float(Vertex.get('X')) for Vertex in Region.iter('Vertex')]).astype('int') #iterate each vertex
            y = np.array([float(Vertex.get('Y')) for Vertex in Region.iter('Vertex')]).astype('int')
            objid = np.array([int(Region.get('Id'))])
            classname = np.array([classluts[idx]])
            df = pd.DataFrame({'classname': classname,
                               'objid': objid,
                               'x': [x],
                               'y': [y], })
            dfs.append(df)

    dff = pd.concat(dfs).reset_index(drop=True)

    slide = OpenSlide(os.path.join(imsrc,imfn))
    rgb_dim = slide.dimensions
    target_level = slide.get_best_level_for_downsample(20)
    target_dim = slide.level_dimensions[target_level]
    rsf = [x/y for x,y in zip(rgb_dim,target_dim)][0]
    TA = slide.read_region(location=(0,0),level=target_level,size=slide.level_dimensions[target_level])
    TA = np.array(TA)
    bw = (150 < np.array(TA)[:, :, 0]) & (np.array(TA)[:, :, 1] < 210)
    bw = closing(bw, square(3))
    minTA = 10000
    bw = remove_small_objects(bw, min_size=minTA, connectivity=2)
    minTAhole = 4000
    bw = remove_small_holes(bw, area_threshold=minTAhole)
    # bw = clear_border(bw)
    TA = np.sum(bw)

    # slide = OpenSlide(os.path.join(imsrc,imfn))
    # rsf = 10 #8um = 1.25x #4um = 2.5x, #2um=5x, 1um=10x, 0.5um=20x, 0.25um=40x
    # rsf = rsf/float(slide.properties['openslide.mpp-x'])
    # target_dim = slide.dimensions
    # target_dim = [round(np.ceil(_/rsf)) for _ in imdim]

    mask = np.zeros(target_dim[::-1], dtype = np.uint8) #white
    for idx,elem in dff.iterrows():
        contours = np.array([elem['x'],elem['y']])
        contours2 = (contours/rsf).astype(int)
        mask = cv2.fillPoly(mask, pts=[contours2.T], color=idx+1)

    #save roi mask
    Image.fromarray(mask.astype('int8')).save(dstfn)

    #save TA mask
    Image.fromarray(bw.astype('int8')).save(TAdstfn)

    ROIA = np.sum(mask)
    ratio = round(ROIA/TA*100)
    # except:
    #     ratio = 0
    #     ROIA = 0
    #     TA = 0
    print('create roi for ',fn,'_elapsed sec:',round(time()-start))
    return [fn,ROIA,TA,ratio]