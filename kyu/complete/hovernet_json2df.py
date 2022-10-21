
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
from DLcomposition import DLcomposition
# from dl2distancemap import dl2distancemap
from dl2distancemap_byroiv2 import dl2distancemap_byroiv2
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
    return np.max((MA,ma)),np.min((MA,ma)),orientation

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
    #special cases
    if celltype == 1: celltype = 2 #corneum cells are epidermal spinousum
    if celltype == 12: celltype =10
    return celltype

def isinroi(cnt,roi,rsfw_ndpi2roi,rsfh_ndpi2roi):
    inroi = roi.getpixel((cnt[0]//rsfw_ndpi2roi,cnt[1]//rsfh_ndpi2roi))
    return inroi

def find_resident_area(tissueid, sectionid, dlareas):
    if tissueid == 12: tissueid = 10
    tissueid = tissueid - 1
    return dlareas.loc[sectionid][tissueid]

def find_c2tdist(cnt,roiid,dldist,rsfw_ndpi2roi,rsfh_ndpi2roi):
    distances = [_[int(cnt[1]//rsfh_ndpi2roi),int(cnt[0]//rsfw_ndpi2roi)].astype(np.float32)*4 for _ in dldist[roiid-1]]
    return distances

def hovernet_json2df(jsonsrc,ndpisrc=None,dlsrc=None,roisrc=None):
    classify_cell = True
    mask_roi = True
    dst = os.path.join(os.path.dirname(jsonsrc), 'dfv2')
    if not os.path.exists(dst): os.mkdir(dst)

    jsons = natsorted([_ for _ in os.listdir(jsonsrc) if _.endswith('.json')])
    # jsons = [_ for _ in jsons if not 'duplicate' in _]
    # jsons = [_ for _ in jsons if not '156' in _]
    # jsons = [_ for _ in jsons if not '167' in _]
    jsons = ['37.json']

    # jsons = jsons[::-1]
    pkls = []
    for idxj,jsonnm in enumerate(jsons): #looping only once
        print(idxj,'/',len(jsons))
        #read and format json into dataframe
        imID,ext = os.path.splitext(jsonnm)
        dstfn = os.path.join(dst, '{}.pkl'.format(imID))
        # if os.path.exists(dstfn):
        #     json = pd.read_pickle(dstfn)
        #     pkls.append(json)
        #     continue
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
                print('celltype classified')

            if mask_roi:
                roinm = jsonnm.replace(ext, '_tissue_binary.tif')
                roi = Image.open(os.path.join(roisrc, roinm))
                roiw, roih = roi.size
                rsfw_ndpi2roi = ndpiw / roiw
                rsfh_ndpi2roi = ndpih / roih

                #label and convert back to pillow image
                roiarr = np.array(roi)
                roiarrL = label(roiarr)
                roiimL = Image.fromarray(roiarrL)
                #classify section id for each cell
                json['inroi'] = json['centroid'].apply(lambda centroid: isinroi(centroid, roiimL, rsfw_ndpi2roi, rsfh_ndpi2roi))
                #eliminate cells not in any roi
                json = json[json['inroi'] > 0].reset_index(drop=True)

                # calculate resident area #TO-DO resident area is composition of a section; shouldn't it be area of connected component in which the cell reside?
                dlareas = DLcomposition(roi,dl) #area is confined by roi
                json['resident_area'] = json.apply(lambda x: find_resident_area(x.type, x.inroi, dlareas), axis=1)

                # calculate distance from objects
                start = time()

                # distmap = dl2distancemap(roi,dl) #11 channel dl distance
                distmap = dl2distancemap_byroiv2(roi, dl)  # 11 channel dl distance
                print('calculation time for distance map: ',time()-start)
                # cell to tissue distance
                # need to calculate this by roi
                # json['c2t_distance'] = json['centroid'].apply(lambda centroid: find_c2tdist(centroid, distmap, rsfw_ndpi2roi, rsfh_ndpi2roi))

                json['c2t_distance'] = json.apply(lambda x: find_c2tdist(x.centroid, x.inroi, distmap, rsfw_ndpi2roi, rsfh_ndpi2roi), axis=1)
                # json[['Dcorneum','Dspinosum','Dshaft','Dfollicle','Dmuscle','Doil','Dsweat','Dnerve','Dblood','Decm','Dfat']] = pd.DataFrame(json.c2t_distance.tolist())
                print('tissue ID assigned to each cell')

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
        json['dist2nearest'] = distance/2 #divide by two to go from 20x to 1um/px

        json['oriA'] = json['orientation'][indices[:, 1]].reset_index(drop=True)
        json['oriB'] = json['orientation'][indices[:, 2]].reset_index(drop=True)
        json['local_align'] = json[['orientation', 'oriA', 'oriB']].std(axis=1) / json[
            ['orientation', 'oriA', 'oriB']].mean(axis=1)
        print('saved : ', dstfn)
        json.to_pickle(dstfn)
        pkls.append(json)

if __name__ == "__main__":
    jsonsrc = r'\\fatherserverdw\Q\research\images\skin_aging\wsi\hovernet_out\json'
    dlsrc = r'\\fatherserverdw\Q\research\images\skin_aging\1um\classification_v9_combined'
    roisrc = r'\\fatherserverdw\Q\research\images\skin_aging\annotation\roi\tif'
    ndpisrc = r'\\fatherserverdw\Q\research\images\skin_aging\wsi'
    hovernet_json2df(jsonsrc,ndpisrc,dlsrc,roisrc)