# read tif - scale up - save as ome-tiff
from time import time
import pyvips
import os

def tif2ometiff(impth,rsf=1):
    start = time()
    imobj = pyvips.Image.new_from_file(impth)
    # resize image
    imobj = imobj.resize(rsf,kernel='nearest')
    if imobj.hasalpha(): imobj = imobj[:-1]
    # split grayscale to zstack of binary


    image_height = imobj.height
    image_width = imobj.width
    image_bands = imobj.bands
    imobj = imobj.copy()
    imobj.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    imobj.set_type(pyvips.GValue.gstr_type, "image-description",
                   f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <!-- Minimum required fields about image dimensions -->
            <Pixels DimensionOrder="XYCZT"
                    ID="Pixels:0"
                    SizeC="{image_bands}"
                    SizeT="1"
                    SizeX="{image_width}"
                    SizeY="{image_height}"
                    SizeZ="1"
                    Type="uint8">
            </Pixels>
        </Image>
    </OME>""")
    end = time()
    print('elapsed {} sec'.format(round(end-start)))
    return imobj


if __name__ == '__main__':
    src = '/Volumes/Digital pathology image lib/JHU/Laura Wood/BTC project/230501 BTC patient002/DLTL run1/DLTLprocess_single/ImAnnotationbyCNN_run2'
    imnm = 'z-0053_2023-03-30 13.48.54_DLAnnMap_1.tif'
    outnm = imnm.replace('tif','ome.tiff')

    impth = os.path.join(src,imnm)
    ometiff = tif2ometiff(impth,rsf=4)

    #Compression Types: jpeg,jp2k,lzw
    # choose jpeg to save space
    # choose lzw for loseless compression of tissue segmentation map
    # don't use jp2k. its behavior is a bit wierd for now
    ometiff.tiffsave(os.path.join(src,outnm), compression="lzw", tile=True,
                   tile_width=512, tile_height=512,
                   pyramid=True, subifd=True)
