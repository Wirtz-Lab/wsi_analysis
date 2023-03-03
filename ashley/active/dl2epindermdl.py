from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import numpy as np
from skimage.morphology import closing, square, remove_small_objects, remove_small_holes
import os
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import ndimage
from skimage.measure import label, regionprops, regionprops_table
import scipy


def replace_dermis_pixel_with_ring(obj,derm):  # obj is a boolean object
    obj_dilate = ndimage.binary_dilation(obj).astype('int')
    ring = obj_dilate - np.array(obj).astype('int')

    labeled_ring = derm * ring
    if not (np.sum(labeled_ring) == 0):
        count = np.bincount(np.ndarray.flatten(labeled_ring))
        replace_val = np.argmax(count[1:]) + 1
        [x, y] = np.where(obj)
        derm[x, y] = replace_val

    return derm

# pass in individual follicle object, not the entire mask
def is_follicle(obj,id, thres_derm4_area,round_threshold, props):
    #too small
    #not round
    if(np.sum(obj.area) < thres_derm4_area):
        return False

    #if it's not too small,check shape
    x,y,w,h = cv2.boundingRect(props[id].coords)
    rect_area = w*h
    extent = obj.area / rect_area
    if(extent > round_threshold):
        return False

    return True


# param dl : pass in an image object with Image.open(os.path.join(dlcropsrc, img_name))
def dlcorrection(dl):
    dl_arr = np.array(dl)
    dl_arr[dl_arr == 12] = 0
    col = (dl_arr == 10)  # collagen

    # 2.remove small object and fill holes in collagen to make it a single connected body
    minTA = 60000
    minTAhole = 10000
    # resize to expedite
    (width, height) = (dl.width // 10, dl.height // 10)
    collagen = Image.fromarray(col).resize((width, height), resample=0)
    collagen = closing(collagen, square(3))  # 13sec
    collagen = remove_small_objects(collagen, min_size=minTA, connectivity=2)  # 6sec
    collagen = remove_small_holes(collagen, area_threshold=minTAhole).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(collagen.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)  # 2sec
    collagen = cv2.dilate(opening, kernel, iterations=3)
    collagen = cv2.dilate(opening, kernel, iterations=3)
    # resize back
    collagen = Image.fromarray(collagen).resize((dl.width, dl.height), resample=0)
    collagen_arr = np.array(collagen)
    print('removed small objects in collagen')
    # 3. find epidermal-dermis junction
    [xt, yt] = np.where(collagen_arr)
    # we actually want to find the minimum x-coordinate (because this image is weirdly oriented)
    unique_y = np.unique(yt)
    min_x = [np.amin(np.where(collagen_arr[:, y])) for y in unique_y]
    thresh = 800  # i'm not sure that this will work with every image. TODO: change
    hist, bins = np.histogram(min_x)
    # get the value range to remove
    remove_bins = [b for h, b in zip(hist, bins) if h < thresh]
    # new junction, remove parts with low frequency in histogram,use remove_bins[0] because it represents where the junction begins to drop
    # TODO: need to change this to be more robust, what if the junction drops in the middle,
    new_junction_x = [x for x in min_x if x < np.round(remove_bins[0].astype("uint32"))]
    new_junction_y = list(range(len(new_junction_x)))
    print('defined epidermis dermis junction')
    # 2: correct anything that is miss classified in epi
    # get new contour
    collagen_tmp = deepcopy(collagen_arr)
    junction_offset = 750
    for x, y in zip(new_junction_x, new_junction_y):
        collagen_tmp[np.max(new_junction_x) + junction_offset:, y] = 0
    collagen_tmp[:, np.max(new_junction_y):] = 0
    max_cx = max(new_junction_x)
    max_cy = max(new_junction_y)
    collagen_tmp = ndimage.binary_fill_holes(collagen_tmp).astype(collagen_tmp.dtype)
    contours, hierarchy = cv2.findContours(collagen_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #draw contour of junction
    img = np.zeros((9746, 11162))
    dontPrint = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)  # 100
    cv2.fillPoly(img, contours, (255, 0, 0))
    junction_mask = np.array(img)
    img1 = img.astype(bool)
    img1[max_cx + 1:, :] = 0
    [cx, cy] = np.where(img1)
    # epidermis-dermis junction is not accurate, extend the junction by 750 pix
    epi = deepcopy(dl_arr)
    # epi[:,np.max(cy)+1:] = 0
    epi[np.max(cx) + 1:, :] = 0
    [x, y] = np.where(junction_mask)
    epi[x + junction_offset, y] = 0

    # iterate through epi and replace value with 1 or 2, ignore 0 and 12
    window_size = 20
    print('iterating through epidermis to replace non-epidermal components to corneum or spinosum')
    # takes around 38 seconds
    for x in range(np.max(x)):
        for y in range(epi.shape[1]):  # max of y
            if not (epi[x, y] == 0 or epi[x, y] == 1 or epi[x, y] == 2 or epi[x, y] == 12 or epi[
                x, y] == 10):  # it can have some 10 collagen in junction??????

                # question: is it better to do a window search or dilate and use the ring to calculate
                # replace pixel value with 1 or 2
                window_shape = [(x - window_size) if (x - window_size) >= 0 else 0,
                                (x + window_size), y + window_size,
                                (y - window_size) if (y - window_size) >= 0 else 0]  # L,R,T,B

                window = epi[window_shape[0]:window_shape[1], window_shape[3]:window_shape[2]]
                # check pixel values in window
                count1 = np.sum([window == 1])
                count2 = np.sum([window == 2])

                if (count1 >= count2):
                    epi[x, y] = 1
                else:
                    epi[x, y] = 2


    # initialize new_dl_arr and make changes to it, do not modify dl_arr
    new_dl_arr = deepcopy(dl_arr)
    # change new_dl_arr based on new epi
    [xt, yt] = np.where(epi == 1)
    new_dl_arr[xt, yt] = 1
    [xt, yt] = np.where(epi == 2)
    new_dl_arr[xt, yt] = 2

    # initialize mask for dermis
    derm = deepcopy(new_dl_arr)
    [x, y] = np.where(epi)
    derm[x, y] = 0

    # replace any 1, 2 in dermis
    derm1 = (derm == 1)  # replace by 3
    derm2 = (derm == 2)  # replace by 4
    if (np.sum(derm1) != 0):
        # replace with 3
        [x, y] = np.where(derm1)
        derm[x, y] = 3

    if (np.sum(derm2) != 0):
        # replace with 4
        [x, y] = np.where(derm2)
        derm[x, y] = 4

    derm3 = (derm == 3)
    thres = 850  # is this too big

    # replace all derm3 with collagen, then add the new mask back
    [x, y] = np.where(derm3)
    derm[x, y] = 10

    #3 should have nothing else in it,should be touching four
    if not np.sum(derm3) == 0:
        derm3 = ndimage.binary_fill_holes(derm3).astype(int)
        derm3 = remove_small_objects(derm3, thres)
        # dilate derm3 to create a ring around derm3, check if the rings contain 4
        # if not, remove everything in that ring
        label_derm = label(derm3)
        # this step eliminates any 3 not touching 4
        # for i in np.unique(label_derm)[1:]: #0 is background
        for i in range(1, np.max(label_derm)):
            # if too small, replace with collagen
            if (np.sum(label_derm == i) < thres):
                [x, y] = np.where(label_derm == i)
                derm3[x, y] = 0
            else:

                ring = ndimage.binary_dilation(label_derm == i).astype('bool')
                ring = ring * derm
                tmp = (ring) & (derm == 4)  # check if 3 is touching 4

                if np.sum(tmp) == 0:
                    # TODO: change pixel value to the majority of the ring
                    replace_dermis_pixel_with_ring(label_derm == i)
                else:
                    [x, y] = np.where(label_derm == i)
                    derm3[x, y] = 0
                    derm[x, y] = 3
    # update derm with new derm 3
    [x1, y1] = np.where(derm3)
    derm[x1, y1] = 3

    # 4 should be big enough, should only have 3 and 6 inside
    thres_derm4_area = 2000    #how big: above 2000px (above 20 cells)
    round_threshold = 0.9

    # smooth derm4
    derm4 = (derm == 4)
    derm4 = cv2.morphologyEx(derm4.astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=2)  # use the same kernel as collagen

    if not np.sum(derm4) == 0:
        labels = label(derm4)
        props = regionprops(labels)
        # fill hair follicle to create query points
        # query ogDLmask to see if non3|6 exists

        for id, prop in enumerate(props):  # id must +1
            if not is_follicle(prop, id,thres_derm4_area,round_threshold,props):
                # replace using ring
                prop_dilate = ndimage.binary_dilation(labels == (id + 1)).astype('bool')
                ring = prop_dilate ^ (labels == (id + 1))
                labeled_ring = ring * derm
                count = np.bincount(np.ndarray.flatten(labeled_ring))
                replace_val = np.argmax(count[:])  # could replace with zero??
                [x, y] = np.where(labels == (id + 1))  # error
                derm[x, y] = replace_val

            else:  # check if 4 only has 3 and 6 inside
                filled_prob = ndimage.binary_fill_holes(labels == (id + 1)).astype(int)
                valuesinhair = derm * filled_prob

                thingstoremove = (valuesinhair != 3) | (valuesinhair != 6) | (valuesinhair != 4)

                if (np.sum(thingstoremove.astype('int')) != 0):  # TODO: fix?
                    derm = replace_dermis_pixel_with_ring(thingstoremove,derm)


        # 5 should not have anything inside it, 5 should only touch 12 , 10
        thres_derm5_area = 2000
        derm5 = (derm == 5)
        derm5 = ndimage.binary_fill_holes(derm5).astype(int)  # fill 5, don't forget to update derm
        labels = label(derm5)
        props = regionprops(labels)
        # update derm with filled derm5
        [x, y] = np.where(derm5)
        derm[x, y] = 5
        for id, prop in enumerate(props):  # id must +1
            if prop.area < thres_derm5_area:
                # replace using ring
                replace_dermis_pixel_with_ring(labels == (id + 1))
            else:  # 5 must not have anything inside, 5 should only touch 12 (which is 0) ,10
                # TODO: how to replace entire tissue section around 5 ?
                prop_dilate = ndimage.binary_dilation(labels == (id + 1)).astype('bool')
                ring = prop_dilate ^ (labels == (id + 1))
                labeled_ring = ring * derm
                replace_dict = {8: 5, 11: 0, 9: 10}  # replace pixel 8 with 5, ...
                # change 8 to 5, 11 to 0 (12), and 9 to 10
                for key in replace_dict:
                    label_key = label(derm == key)
                    [xk, yk] = np.where(labeled_ring == key)
                    if not (len(xk) == 0):  # i.e. if xk is not empty list
                        [x, y] = [xk[0], yk[0]]
                        to_change_lb = label_key[x, y]
                        to_change_area = (label_key == to_change_lb)
                        [cx, cy] = np.where(to_change_area)
                        derm[cx, cy] = replace_dict[key]

        # 6 oilgland, 7 sweatgland, 8 nerve, 9 bloodvessel should not contain  any other class within it
        # thres is 850
        to_fill_id = [6, 7, 8, 9]
        for id in to_fill_id:
            if (np.sum(derm == id) != 0):
                # remove small objects
                dermid = (derm == i)
                [x, y] = np.where(dermid)
                derm[x, y] = 10
                tmp_derm = remove_small_objects(dermid, thres).astype(np.uint8)

                # fill
                tmp_derm = ndimage.binary_fill_holes(tmp_derm).astype(int)
                [x, y] = np.where(tmp_derm)
                derm[x, y] = id

        [x, y] = np.where(derm == 0)
        derm[x, y] = 12  # 12 is white
        [x, y] = np.where(epi == 0)
        epi[x, y] = 12  # 12 is white

        # put dermis on top of epi
        final_img = deepcopy(epi)
        # 3-12
        for i in range(3, 12):
            dermtmp = (derm == i)
            [x, y] = np.where(dermtmp)
            final_img[x, y] = i

        return Image.fromarray(final_img)

if __name__ == "__main__":
    dlcropsrc = r'\\fatherserverdw\kyuex\analysis output\datadst\20220929\dlcrop'
    img_name = '2022-06-08 18.13.05sec2.png'
    dl = Image.open(os.path.join(dlcropsrc, img_name))
    correcteddl = dlcorrection(dl)

    dst = os.path.join(dlcropsrc,'corrected230303')
    if not os.path.exists(dst):os.mkdir(dst)
    correcteddl.save(os.path.join(dst,img_name))














