import os
import pandas as pd
from natsort import natsorted
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from openslide import OpenSlide
import numpy as np
import cv2
def cntarea(cnt):
    cnt = np.array(cnt)
    area = cv2.contourArea(cnt)
    return area
def cntAR(cnt):
    cnt = np.array(cnt)
    #Orientation, Aspect_ratio
    (x,y),(MA,ma),orientation = cv2.fitEllipse(cnt)
    aspect_ratio = MA/ma
    return aspect_ratio
def cntsol(cnt):
    cnt = np.array(cnt)
    #Solidity
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return solidity
def cntExtent(cnt):
    cnt = np.array(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent
def cntEquiDia(cnt):
    cnt = np.array(cnt)
    #Equi Diameter
    area = cv2.contourArea(cnt)
    equi_diameter = np.sqrt(4*area/np.pi)
    return equi_diameter

def cellclass(cnt,dl,rsfw_ndpi2dl,rsfh_ndpi2dl):
    celltype = dl.getpixel((cnt[0]//rsfw_ndpi2dl,cnt[1]//rsfh_ndpi2dl))
    return celltype

def hovernet_json2df(jsonsrc,ndpisrc=None,dlsrc=None):
    classify_cell = True

    dst = os.path.join(os.path.dirname(jsonsrc), 'df')
    if not os.path.exists(dst): os.mkdir(dst)

    jsons = natsorted([_ for _ in os.listdir(jsonsrc) if _.endswith('json')])
    for jsonnm in jsons:
        #read and format json into dataframe
        imID,ext = os.path.splitext(jsonnm)
        dstfn = os.path.join(dst, '{}.pkl'.format(imID))
        if os.path.exists(dstfn): continue
        json = os.path.join(jsonsrc, jsonnm)
        try:
            json = pd.read_json(json, orient='index')
        except:
            continue
        json = pd.DataFrame(json[0].loc['nuc']).T.reset_index(drop=True).drop(columns=['type_prob'])

        if (dlsrc is not None) & (ndpisrc is not None):
            #calculate rescale factor between ndpi and dlmask
            ndpinm = jsonnm.replace(ext,'.ndpi')
            ndpi = OpenSlide(os.path.join(ndpisrc, ndpinm))
            ndpiw, ndpih = ndpi.dimensions

            dlnm = jsonnm.replace(ext, '.tif')
            dl = os.path.join(dlsrc, dlnm)
            dl = Image.open(dl)
            dlw, dlh = dl.size

            rsfw_ndpi2dl = ndpiw / dlw
            rsfh_ndpi2dl = ndpih / dlh

        # query centroid on tissue map to obtain tissue component ID where the cell is contained
            if classify_cell:
                json['type'] = json['centroid'].apply(lambda row: cellclass(row,dl,rsfw_ndpi2dl,rsfh_ndpi2dl))
        json = json[json['contour'].map(len) > 5]

        json['Area'] = json['contour'].apply(lambda row: cntarea(row))
        json['AR'] = json['contour'].apply(lambda row: cntAR(row))
        json['Sol'] = json['contour'].apply(lambda row: cntsol(row))
        json['Extent'] = json['contour'].apply(lambda row: cntExtent(row))
        json['EquiDia'] = json['contour'].apply(lambda row: cntEquiDia(row))

        json['imID'] = [imID]*len(json)
        json.to_pickle(dstfn)

if __name__ == "__main__":
    jsonsrc = r'\\fatherserverdw\Q\research\images\skin_aging\wsi\hovernet_out\json'
    dlsrc = r'\\fatherserverdw\Q\research\images\skin_aging\1um\classification_v9_combined'
    ndpisrc = r'\\fatherserverdw\Q\research\images\skin_aging\wsi'
    hovernet_json2df(jsonsrc,ndpisrc,dlsrc)