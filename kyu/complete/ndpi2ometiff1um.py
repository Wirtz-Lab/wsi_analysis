import pyvips
import openslide
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from natsort import natsorted

def svs2tiff(svs,rsf,svs_dst):
    if not os.path.exists(svs_dst): os.mkdir(svs_dst)
    src,fn = os.path.split(svs)
    fn,ext = os.path.splitext(fn)
    fn1 = fn + '.ome.tiff'
    if os.path.exists(os.path.join(svs_dst,fn1)):
        print('exists')
        return
    print('processing: ',fn)
    svs_obj = openslide.OpenSlide(svs)
    try:
        svs_img = svs_obj.read_region(location=(0,0),level=0,size=svs_obj.level_dimensions[0]).convert('RGB')
        print('opened image: ',fn)
    except:
        print('OOM: ',fn)
        return

    resize_factorx = rsf/float(svs_obj.properties['openslide.mpp-x']) #8um = 1.25x #4um = 2.5x, #2um=5x, 1um=10x, 0.5um=20x, 0.25um=40x
    resize_factory = rsf/float(svs_obj.properties['openslide.mpp-y'])

    resize_dimension = tuple([int(np.ceil(svs_obj.dimensions[0]/resize_factorx)),int(np.ceil(svs_obj.dimensions[1]/resize_factory))])
    svs_img = svs_img.resize(resize_dimension,resample=Image.Resampling.NEAREST)
    print('resized image: ',fn)

    # pyvips
    svs_img = pyvips.Image.new_from_array(obj = svs_img)
    if svs_img.hasalpha():
        svs_img = svs_img[:-1]
    image_height = svs_img.height
    image_width = svs_img.width
    image_bands = svs_img.bands

    # split to separate image planes and stack vertically ready for OME
    svs_img = pyvips.Image.arrayjoin(svs_img.bandsplit(), across=1)

    # set minimal OME metadata
    # before we can modify an image (set metadata in this case), we must take a
    # private copy
    svs_img = svs_img.copy()
    svs_img.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    svs_img.set_type(pyvips.GValue.gstr_type, "image-description",
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

    #JPEG,JP2K,LZW,
    svs_img.tiffsave(os.path.join(svs_dst,fn1), compression="jp2k", tile=True,
                     tile_width=512, tile_height=512,
                     pyramid=True, subifd=True)
    print("Image sucessfully saved!")

if __name__ == '__main__':
    input_dir = '/Volumes/Kyu/unstain2stain/unstain2stain_wsi/HE'
    output_dir = os.path.join(input_dir,'1um')
    filenames = [x for x in os.listdir(input_dir) if x.endswith(".ndpi")]
    # filenames = natsorted(filenames)
    filenames = sorted(filenames, key=lambda x: os.stat(os.path.join(input_dir, x)).st_size) #sort by size
    # Loop through all the files in the input directory
    # for filename in tqdm(filenames,total=len(filenames),desc='Processed ndpis:',colour='red'):
    for filename in filenames[::-1]:
        input_filepath = os.path.join(input_dir, filename)
        svs2tiff(input_filepath, 1, output_dir)