from PIL import Image
import os
import glob
import numpy as np
from skimage import morphology
from sklearn.cluster import KMeans

src = r'\\fatherserverdw\Q\research\images\CLUE\3D study\he\1.25x'
dst = os.path.join(src,'TA')
if not os.path.exists(dst): os.mkdir(dst)
images = glob.glob(os.path.join(src,'*jpg'))

for image in images:
    imobj = Image.open(image)
    imarr = np.array(imobj)
    data = imarr[:, :, 2].ravel()
    kmeans = KMeans(n_clusters=2).fit(data.reshape(-1, 1))
    kmeans.predict(data.reshape(-1, 1))
    threshold_value = kmeans.cluster_centers_[0] - (kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) / 4
    immsk = np.zeros_like(imarr[:, :, 0])
    immsk[imarr[:, :, 2] < threshold_value] = 1
    immsk2 = morphology.binary_closing(immsk)
    immsk3 = morphology.area_opening(immsk2, area_threshold=100000)
    immsk4 = morphology.area_closing(immsk3, area_threshold=1000)
    base, fn = os.path.split(image)
    fn, ext = os.path.splitext(fn)
    dstfn = os.path.join(dst, fn + '.png')
    Image.fromarray((immsk4 * 255).astype('uint8')).save(dstfn)