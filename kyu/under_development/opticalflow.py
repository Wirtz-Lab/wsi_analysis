from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

def opticalflow(impthA,impthB):
    print(impthA,impthB)
    ref=np.array(Image.open(impthA))
    mov=np.array(Image.open(impthB))
    refg=rgb2gray(ref)
    movg=rgb2gray(mov)
    v, u = optical_flow_tvl1(refg, movg)
    nr, nc = refg.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),indexing='ij')
    transmap = np.array([row_coords + v, col_coords + u])
    outputs =[]
    for i in range(3):
        mov_img_warp = warp(mov[:,:,i],transmap ,mode='edge')
        outputs.append(mov_img_warp)
    return Image.fromarray(np.stack([(_*255).astype(np.uint8) for _ in outputs],axis=2))