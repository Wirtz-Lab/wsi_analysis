import pandas as pd
from skimage.measure import label
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import numpy as np

def DLcomposition(roi,dl):
    # Input: roi, dl pillow images
    roi = roi.resize(dl.size)
    roiarr = np.array(roi)
    roiarrL = label(roiarr)
    dlarr = np.array(dl)
    df = []
    for roi in np.unique(roiarrL):
        dltmp = dlarr * (roiarrL == roi)
        dlareas = np.histogram(dltmp, bins=range(14))
        dlareas = dlareas[0]
        dlareas = dlareas.tolist()
        df.append([dlareas[0:10] + [dlareas[10] + dlareas[12]] + [dlareas[11]]])
    df = pd.DataFrame(np.squeeze(df))
    # df.rename(columns={0: 'corneum', 1: 'spinosum', 2: 'shaft', 3: 'follicle', 4: 'muscle', 5: 'oil',
    #                    6: 'sweat', 7: 'nerve', 8: 'blood', 9: 'ecm', 10: 'fat'}, inplace=True)
    return df

