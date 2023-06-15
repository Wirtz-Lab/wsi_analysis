import pyvips  #must use conda to install
import os
from tqdm import tqdm
from natsort import natsorted

def ims2ometiff(input_dir,output_dir):
    ims = [x for x in os.listdir(input_dir) if x.endswith(".tif")]
    ims = natsorted(ims)
    imobjs = []
    for im in ims:
        impth = os.path.join(input_dir, im)
        imobj = pyvips.Image.new_from_file(impth)
        # imobj = pyvips.Image.openslideload(impth,level=4)
        if imobj.hasalpha(): imobj = imobj[:-1]
        imobjs.append(imobj)
    print('z stack height :',len(imobjs))
    comp = pyvips.Image.arrayjoin(imobjs, across=1)
    image_height = imobj.height
    image_width = imobj.width
    image_bands = imobj.bands
    comp = comp.copy()

    # set minimal OME metadata
    # before we can modify an image (set metadata in this case), we must take a
    # private copy
    comp.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    comp.set_type(pyvips.GValue.gstr_type, "image-description",
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
                    SizeZ="{len(imobjs)}"
                    Type="uint8">
            </Pixels>
        </Image>
    </OME>""")

    outfn = 'stack_kyu.ome.tiff'
    #jpeg,jp2k,lzw,
    print('writing file')
    # jp2k breaks the format somehow
    comp.tiffsave(os.path.join(output_dir, outfn), compression="jpeg", tile=True,
                  tile_width=512, tile_height=512,
                  pyramid=True, subifd=True)
    print(os.path.join(output_dir, outfn))

if __name__=='__main__':
    input_dir = '/Volumes/Digital pathology image lib/JHU/Laura Wood/IF for SenPan001/FWPanc001_60/1st round/well image stitch/1x_'
    # output_dir = os.path.join(input_dir, 'zstack')
    # if not os.path.exists(output_dir): os.mkdir(output_dir)
    ims2ometiff(input_dir,input_dir)
