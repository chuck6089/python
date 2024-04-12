# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:33:02 2024

@author: xzd6089
"""


import os
import numpy as np
import cv2
from external.prometric.pmxm_reader_utils import pmxm_reader_utils
import imageio

def convert2img(array):
    array[array < 0] = 0
    array_max = np.max(array)
    image = np.uint16(array*(65535/array_max))
    #cv2.imwrite('tempimage.png',image)
    #Imagetosave = PIL.fromarray(image)
    imageio.imsave('tempimage.png',image)
    image = imageio.imread('tempimage.png')   #this is to avoid a weird bug by cv2.rectangle, the image has to be reloaded in order to plot
    return image


# %%
pmxm_reader = pmxm_reader_utils()

pmxmpath = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75014\Flare_March2024\SCS39\Lumus_SN75014_SCS39_2mm_Lumus_Left_Primary_NoProjectorAperture_20240321.pmxm'

listofmeasurement = pmxm_reader.get_meas_list(pmxmpath)

length = len(listofmeasurement)

for meas in listofmeasurement:
    measure_ID = meas[0]
    keyword = meas[1]
    XYZ = pmxm_reader.get_xyz_arrays(pmxm_file = pmxmpath, meas_id = measure_ID)
    image = convert2img(XYZ[1])
    path, filename = os.path.split(pmxmpath)
    imageio.imsave(path + "\\" + keyword + ".png",image)