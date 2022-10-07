import pandas as pd
from skimage.measure import label
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import numpy as np

def DLcomposition(dl):
    # Input: roi, dl pillow images
    dl_img = Image.open(dl)
    dlarr = np.array(dl_img)
    dlareas = np.histogram(dlarr, bins=range(np.max(dlarr)+2))
    dlareas = dlareas[0]
    dlareas = dlareas[1:13]
    return dlareas

