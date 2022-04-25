def get_im(pth,nm):
    pthTA = join(pth,'TA')
    if not exists(pthTA):
      makedirs(join(pth,'TA'))
    im = Image.open(join(pth,nm))

    if exists(join(pthTA,nm[:-3]+'png')):
        TAnm = listdir(TApth)[zc]
        TA=Image.open(join(TApth,TAnm))
    else
        TA = find_tissue_area(im, nm) # need to make find_tissue_area()
        imwrite(TA, join(pthTA,nm[:-3]+'png'))

    return im, TA