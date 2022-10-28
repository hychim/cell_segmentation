import numpy as np
import time, os, sys
from urllib.parse import urlparse
from PIL import Image
import cv2
from cellpose import models, io

# import required module
import os
from pathlib import Path
# assign directory
directory = '/mnt/external-images-pvc/david/liveCellPainting/raw_data/exp156/Images/'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if filename.endswith(".tiff"):
        img = io.imread(f)
        
        # cellpose
        model = models.Cellpose(gpu=True, model_type='cyto')
        channels = [[0,0]]

        img = io.imread(f)
        masks, flows, styles, diams = model.eval(img, diameter=None, channels=channels)    
        
        im = Image.fromarray(masks)
        im.save("masks/" + Path(f).stem + "_mask.png")