## needs edit
def find_tissue_area(im0):
    imarr = np.array(im0)
    data = imarr[:, :, 2].ravel()
    kmeans = KMeans(n_clusters=2).fit(data.reshape(-1, 1))
    kmeans.predict(data.reshape(-1, 1))
    threshold_value = kmeans.cluster_centers_[0] - (kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) / 4
    immsk = np.zeros_like(imarr[:, :, 0])
    immsk[imarr[:, :, 2] < threshold_value] = 1
    immsk2 = morphology.binary_closing(immsk)
    immsk3 = morphology.area_opening(immsk2, area_threshold=100000)
    immsk4 = morphology.area_closing(immsk3, area_threshold=1000)
    base, fn = split(im0)
    fn, ext = splitext(fn)
    dstfn = join(dst, fn + '.png')
    Image.fromarray((data * 255).astype('uint8'))

    return immsk4