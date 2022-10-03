import pandas as pd
from skimage.measure import label
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import numpy as np

def DLcomposition(dl):
    # Input: roi, dl pillow images
    dl_img = Image.open(dl)
    dlarr = np.array(dl_img)
    df = []
    dlareas = np.histogram(dlarr, bins=range(np.max(dlarr)+2))
    dlareas = dlareas[0]
    dlareas = dlareas.tolist()
    df.append(dlareas[1:13])
    df = pd.DataFrame(np.squeeze(df))

    #for roi in np.unique(roiarrL):
        #dltmp = dlarr * (roiarrL == roi)
        #dlareas = np.histogram(dltmp, bins=range(14))
        #dlareas = dlareas[0]
        #dlareas = dlareas.tolist()
        #df.append([dlareas[0:10] + [dlareas[10] + dlareas[12]] + [dlareas[11]]])
    #df = pd.DataFrame(np.squeeze(df))
    # df.rename(columns={0: 'corneum', 1: 'spinosum', 2: 'shaft', 3: 'follicle', 4: 'muscle', 5: 'oil',
    #                    6: 'sweat', 7: 'nerve', 8: 'blood', 9: 'ecm', 10: 'fat'}, inplace=True)
    return df

