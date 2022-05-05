import os
import pandas as pd
from natsort import natsorted
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from openslide import OpenSlide
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from skimage.measure import label
from time import time

def cntarea(cnt):
    cnt = np.array(cnt)
    area = cv2.contourArea(cnt)
    return area

def cntperi(cnt):
    cnt = np.array(cnt)
    perimeter = cv2.arcLength(cnt,True)
    return perimeter

def cntMA(cnt):
    cnt = np.array(cnt)
    #Orientation, Aspect_ratio
    (x,y),(MA,ma),orientation = cv2.fitEllipse(cnt)
    return MA,ma,orientation

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

def isinroi(cnt,roi,rsfw_ndpi2roi,rsfh_ndpi2roi):
    inroi = roi.getpixel((cnt[0]//rsfw_ndpi2roi,cnt[1]//rsfh_ndpi2roi))
    return inroi

def hovernet_json2df(jsonsrc,ndpisrc=None,dlsrc=None,roisrc=None):
    classify_cell = True
    mask_roi = True
    dst = os.path.join(os.path.dirname(jsonsrc), 'df')
    if not os.path.exists(dst): os.mkdir(dst)

    jsons = natsorted([_ for _ in os.listdir(jsonsrc) if _.endswith('.json')])
    jsons = [_ for _ in jsons if not 'duplicate' in _]
    pkls = []
    for jsonnm in jsons:
        #read and format json into dataframe
        imID,ext = os.path.splitext(jsonnm)
        dstfn = os.path.join(dst, '{}.pkl'.format(imID))
        print(dstfn)
        if os.path.exists(dstfn):
            json = pd.read_pickle(dstfn)
            pkls.append(json)
            continue
        json = os.path.join(jsonsrc, jsonnm)
        try:
            json = pd.read_json(json, orient='index')
        except:
            print('error')
            continue
        json = pd.DataFrame(json[0].loc['nuc']).T.drop(columns=['type_prob'])
        json = json[json['contour'].map(len) > 5].reset_index(drop=True)

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
            if mask_roi:
                roinm = jsonnm.replace(ext, '_tissue_binary.tif')
                roi = Image.open(os.path.join(roisrc, roinm))
                roiarr = np.array(roi)
                roiimL = label(roiarr)
                roiw, roih = roi.size
                rsfw_ndpi2roi = ndpiw / roiw
                rsfh_ndpi2roi = ndpih / roih
                print(np.unique(roiimL))
                roiimL = Image.fromarray(roiimL)
                json['inroi'] = json['centroid'].apply(lambda row: isinroi(row, roiimL, rsfw_ndpi2roi, rsfh_ndpi2roi))
        json['Area'] = json['contour'].apply(lambda row: cntarea(row))
        json['Perimeter'] = json['contour'].apply(lambda row: cntperi(row))
        json['Circularity'] = 4 * np.pi * json['Area'] / json['Perimeter'] ** 2
        json['MA'] = json['contour'].apply(lambda row: cntMA(row))
        json[['MA', 'ma', 'orientation']] = pd.DataFrame(json.MA.tolist())
        json['AspectRatio'] = json['MA'] / json['ma']
        json['Sol'] = json['contour'].apply(lambda row: cntsol(row))
        json['Extent'] = json['contour'].apply(lambda row: cntExtent(row))
        json['EquiDia'] = json['contour'].apply(lambda row: cntEquiDia(row)) # sqrt(4*Area/pi).
        json['imID'] = [int(imID)]*len(json)

        points = pd.DataFrame(json.centroid.tolist()).astype('int')
        nbrs = NearestNeighbors(n_neighbors=3, metric='euclidean').fit(points)
        distances, indices = nbrs.kneighbors(points)
        distance = distances[:, 1]
        json['dist2nearest'] = distance

        json['oriA'] = json['orientation'][indices[:, 1]].reset_index(drop=True)
        json['oriB'] = json['orientation'][indices[:, 2]].reset_index(drop=True)
        json['local_align'] = json[['orientation', 'oriA', 'oriB']].std(axis=1) / json[
            ['orientation', 'oriA', 'oriB']].mean(axis=1)
        print(dstfn)
        json.to_pickle(dstfn)
        pkls.append(json)
    #
    pkls = pd.concat(pkls, ignore_index=True)
    pkls.to_feather(os.path.join(dst, '2d_skin_hovernet.ftr'))
    pkls=pkls[pkls['inroi']>0]
    pkls.to_feather(os.path.join(dst, '2d_skin_hovernet_inroi.ftr'))

if __name__ == "__main__":
    jsonsrc = r'\\fatherserverdw\Q\research\images\skin_aging\wsi\hovernet_out\json'
    dlsrc = r'\\fatherserverdw\Q\research\images\skin_aging\1um\classification_v9_combined'
    roisrc = r'\\fatherserverdw\Q\research\images\skin_aging\annotation\roi\tif'
    ndpisrc = r'\\fatherserverdw\Q\research\images\skin_aging\wsi'
    hovernet_json2df(jsonsrc,ndpisrc,dlsrc,roisrc)