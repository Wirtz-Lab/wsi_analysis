
import numpy as np
import time
from PIL import Image
import cv2
import os
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk


#%%
from time import time
def _time(f):
    def wrapper(*args,**kwargs):
        start=time()
        r=f(*args,**kwargs)
        end=time()
        print("%s timed %f" %(f.__name__,end-start))
        return r
    return wrapper

#%%
# function to pad images to same size:
def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255,255,255))
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded

pad_images_to_same_size = _time(pad_images_to_same_size)
optical_flow_tvl1 = _time(optical_flow_tvl1)
#%%
#first create registered image of two adjacent images manually:
img_files_path = [_ for _ in os.listdir(r'\\fatherserverdw\Kevin\imageregistration\padded_images') if _.endswith(".png")]
img_files_path_complete = [os.path.join(r'\\fatherserverdw\Kevin\imageregistration\padded_images', x) for x in img_files_path]
img_files_path_1 = [x.replace('.png','') for x in img_files_path]

num = int(len(img_files_path)/2) - 1 #idx = 16, or 17th image
num_plus1 = num + 1 #idx = 17, or 18th image
num_minus1 = num - 1 #idx = 15, or 16th image

start = time()

ref_img_path = img_files_path_complete[num]
mov_img_path = img_files_path_complete[num_plus1]
ref_img = np.array(Image.open(ref_img_path))
mov_img = np.array(Image.open(mov_img_path))

ref_img_g = rgb2gray(ref_img)
mov_img_g = rgb2gray(mov_img)
v, u = optical_flow_tvl1(ref_img_g, mov_img_g)
nr, nc = ref_img_g.shape
row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                     indexing='ij')

end = time()
print("time it took to register: "+  str(end-start) + " seconds")

start = time()

mov_img_warp_ra =[]
for i in range(3):
    mov_img_warp = warp(mov_img[:,:,i], np.array([row_coords + v, col_coords + u]),mode='edge')
    mov_img_warp_ra.append(mov_img_warp)

r = np.array(mov_img_warp_ra[0]*255).astype('uint8')
g = np.array(mov_img_warp_ra[1]*255).astype('uint8')
b = np.array(mov_img_warp_ra[2]*255).astype('uint8')
rgb = np.stack([r,g,b],axis=2)
reg_img = Image.fromarray(rgb)
reg_img.save(r'\\fatherserverdw\Kevin\imageregistration\registered_images\\' + str(img_files_path_1[num_plus1]) + '.png')

end = time()
print("time it took to register: "+  str(end-start) + " seconds")
#%%
# repeat again:
start = time()

ref_img_path = img_files_path_complete[num]
mov_img_path = img_files_path_complete[num_minus1]
ref_img = np.array(Image.open(ref_img_path))
mov_img = np.array(Image.open(mov_img_path))

ref_img_g = rgb2gray(ref_img)
mov_img_g = rgb2gray(mov_img)
v, u = optical_flow_tvl1(ref_img_g, mov_img_g)
nr, nc = ref_img_g.shape
row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                     indexing='ij')
end = time()
print("time it took to register: "+  str(end-start) + " seconds")

start = time()

mov_img_warp_ra =[]
for i in range(3):
    mov_img_warp = warp(mov_img[:,:,i], np.array([row_coords + v, col_coords + u]),mode='edge')
    mov_img_warp_ra.append(mov_img_warp)

r = np.array(mov_img_warp_ra[0]*255).astype('uint8')
g = np.array(mov_img_warp_ra[1]*255).astype('uint8')
b = np.array(mov_img_warp_ra[2]*255).astype('uint8')
rgb = np.stack([r,g,b],axis=2)
reg_img = Image.fromarray(rgb)
reg_img.save(r'\\fatherserverdw\Kevin\imageregistration\registered_images\\' + str(img_files_path_1[num_minus1]) + '.png')

end = time()
print("time it took to register: "+  str(end-start) + " seconds")
#%%
#two for loops to recursively create registered images:
# from middle to end (higher index), idx of 17 to 33, or 18th image to 34th image:
start = time()

for idx in range(num_plus1,len(img_files_path_complete)): #idx = 17 to 34 (not including 34)
    if idx == len(img_files_path_complete) - 1:
        break
    ref_img_path = os.path.join(r'\\fatherserverdw\Kevin\imageregistration\registered_images', img_files_path[idx])
    mov_img_path = img_files_path_complete[idx+1]
    ref_img = np.array(Image.open(ref_img_path))
    mov_img = np.array(Image.open(mov_img_path))

    ref_img_g = rgb2gray(ref_img)
    mov_img_g = rgb2gray(mov_img)
    v, u = optical_flow_tvl1(ref_img_g, mov_img_g)
    nr, nc = ref_img_g.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')
    mov_img_warp_ra =[]
    for i in range(3):
        mov_img_warp = warp(mov_img[:,:,i], np.array([row_coords + v, col_coords + u]),mode='edge')
        mov_img_warp_ra.append(mov_img_warp)
    r = np.array(mov_img_warp_ra[0]*255).astype('uint8')
    g = np.array(mov_img_warp_ra[1]*255).astype('uint8')
    b = np.array(mov_img_warp_ra[2]*255).astype('uint8')
    rgb = np.stack([r,g,b],axis=2)
    reg_img = Image.fromarray(rgb)
    reg_img.save(r'\\fatherserverdw\Kevin\imageregistration\registered_images\\' + str(img_files_path_1[idx+1]) + '.png')

end = time()
print("time it took to register: "+  str(end-start) + " seconds")
#%%
# from middle to beginning (lower index), idx of 15 to 0, or 16th image to 1st image:
start = time()

for idx in range(num_minus1,-1,-1): #idx = 15 to -1 (not including -1)
    # declare destination file name, check if it exists already, continue if so
    dst = r'\\fatherserverdw\Kevin\imageregistration\registered_images'
    dstfn = os.path.join(dst,img_files_path[idx-1])
    if os.path.exists(dstfn):
        continue

    if idx == 0:
        break
    ref_img_path = os.path.join(r'\\fatherserverdw\Kevin\imageregistration\registered_images', img_files_path[idx])
    mov_img_path = img_files_path_complete[idx-1]
    ref_img = np.array(Image.open(ref_img_path))
    mov_img = np.array(Image.open(mov_img_path))

    ref_img_g = rgb2gray(ref_img)
    mov_img_g = rgb2gray(mov_img)
    v, u = optical_flow_tvl1(ref_img_g, mov_img_g)
    nr, nc = ref_img_g.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')
    mov_img_warp_ra =[]
    for i in range(3):
        mov_img_warp = warp(mov_img[:,:,i], np.array([row_coords + v, col_coords + u]),mode='edge')
        mov_img_warp_ra.append(mov_img_warp)
    r = np.array(mov_img_warp_ra[0]*255).astype('uint8')
    g = np.array(mov_img_warp_ra[1]*255).astype('uint8')
    b = np.array(mov_img_warp_ra[2]*255).astype('uint8')
    rgb = np.stack([r,g,b],axis=2)
    reg_img = Image.fromarray(rgb)

    reg_img.save(dstfn)


end = time()
print("time it took to register: "+  str(end-start) + " seconds")

