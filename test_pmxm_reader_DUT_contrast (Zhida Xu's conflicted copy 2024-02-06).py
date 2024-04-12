# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:54:02 2023

@author: xzd6089
"""

# %% import and definitoin

import os
import numpy as np
import matplotlib.pyplot as plt
import image_utils
import cv2
import analyzer_portable as analyzer
from external.prometric.pmxm_reader_utils import pmxm_reader_utils

ix = -1
iy = -1


def convert2img(array, flipX = 0):
    array[array < 0] = 0
    array_max = np.max(array)
    ratio = 65430/array_max
    image = np.uint16(array*ratio)  #must use 16bit image to keep the dynamic range
    cv2.imwrite('tempimage.png',image)
    image = cv2.imread('tempimage.png',cv2.IMREAD_UNCHANGED)   #this is to avoid a weird bug by cv2.rectangle, the image has to be reloaded in order to plot
    if flipX == 1:
        image = cv2.flip(image, 1)   #flip in X direction
    return image, ratio
 
def find_measurementIDs(listofmeasurement, keyword):
    indexarray = []
    for measurement in listofmeasurement:
        if keyword in measurement[1]:
            indexarray.append(measurement[0])
    return indexarray
            
def readXYZ_radiant(path, keyword):
    result = []
    listofmeasurement = pmxm_reader.get_meas_list(path)
    index = find_measurementIDs(listofmeasurement, keyword)
    for ID in index:
        XYZ = pmxm_reader.get_xyz_arrays(pmxm_file = path, meas_id = ID)
        result.append(XYZ)
    return result

# %%

pmxm_reader = pmxm_reader_utils()

#listofmeasurement = pmxm_reader.get_meas_list('G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_SN75017_Capstone_0deg_noND_wiPolar.pmxm')

#xyz = pmxm_reader.get_xyz_arrays(pmxm_file = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_SN75017_Capstone_0deg_noND_wiPolar.pmxm', meas_id = 1)


#slbpath = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75017\Projector\Lumus_SN75017_Projector_NewFixture_woNDfilter_B_withWP.pmxm'
#slbpath = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75017\Projector\Lumus_SN75017_Projector_2mm_X1Y_0.75_ERD0.4_01022024.pmxm'
#slbpath = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75017\Projector\Lumus_SN75017_Projector_noLensAperture_3mmCal_X1Y-1_ERD0.4_01022024.pmxm'
slbpath = r"X:\IQT2-Display\Devices\Lumus\Lumus_SN75010\Projector\Lumus V3_0deg\Lumus_SN75010_Projector_4mm_Lumus v3_ERD0.4_20240130_3mmCalibFile.pmxm"
#slbpath = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\SLB\Lumus_SN75017_Projector_ND2.pmxm'
#slbpath = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75017\Projector\Lumus_SN75017_Projector_noLensAperture_3mmCal_X1Y-1_ERD0.4_01022024.pmxm'
#slbpath = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75017\Projector\Lumus_SN75017_Projector_3mm_X1Y-0.75_ERD0.4_01022024.pmxm'
path = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75010\SCS1\Lumus_SN75010_SCS1_2mm_Lumus v4_PupilPosition1_20240131.pmxm'
#path = 'X:\IQT2-Display\Devices\Lumus\Lumus_SN75017\Capstone\Lumus_SN75017_Capstone_Second_Normal_Eyebox_X-20.045_Y-7.154_Z0.8_aftEdgeBlankened.pmxm'
crosshairtemplatepath = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75017\crosshair_template\crosshair_template_2.png'


dirtory, datafilename = os.path.split(path)
datafilename, extension = os.path.splitext(datafilename)
os.makedirs(dirtory + '\\' + datafilename, exist_ok = True)

pupilnumber = analyzer.find_string_between(datafilename,'Position','_')

pupilnumber = '1'

listofmeasurement = pmxm_reader.get_meas_list(path)

keyword = 'Green Neg' #ignore this

aperture_size = 2  #aperture diameter in mm
eyebox_size = 146.8519  #eyebox size in unit of mm^2


center_m_slb = 824
center_n_slb = 523
rotate = 0.5 #deg positive for counter-clockwise
rotate_slb = -26 #deg
#rotate_slb = 26 #deg
squaresize_m_dut = 38.3
squaresize_n_dut = 38.3
squaresize_m_slb = 38.1
squaresize_n_slb = 38.1

FOVx = 34
FOVy = 34
pixperdeg = 18.75



# %% run this section
colorword = 'Green'
Side = 'R'
pupil = int(pupilnumber)
tilt_angle = -5

saddlepoint_size = (16,16)
detectable_saddlepoint = saddlepoint_size[1] - 0

center_m_dut_offset = 5
center_n_dut_offset = 20

if tilt_angle == 0:
    angleword = "0deg"
elif tilt_angle == 5:
    angleword = "+5deg"
elif tilt_angle == -5:
    angleword ="-5deg"
else:
    print("Tilt angle can only be +5deg, 0 deg, -5deg")

xyz = readXYZ_radiant(path, keyword)

Y = xyz[0][1]

data, r = convert2img(Y) 

# crossdata = readXYZ_radiant(path, 'Green Cross')
# crossimage = convert2img(crossdata[0][1])


rows, cols = data.shape

# Create x and y coordinates for the plot
x = np.arange(cols)
y = np.arange(rows)

# Create a meshgrid from x and y
X, Y = np.meshgrid(x, y)

# Plot the 2D profile using a contour plot
plt.contour(X, Y, data, cmap='viridis')
plt.colorbar()  # Add a colorbar for reference
plt.title('2D Profile')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


#template = cv2.imread('crosshair_template_2.png',cv2.IMREAD_GRAYSCALE)
#coords = image_utils.get_crosshairs(crossimage,template)
dist_coef = image_utils.distortion_coefficients("distortion_87deg_v2Lens.xlsx")

undistort,K_mtx = image_utils.undistort(data, dist_coef)
cv2.imwrite('undistort.bmp',undistort)

#center_coord = (np.mean(coords[:,0]),np.mean(coords[:,1]))

#Find contrast~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

current_directory = os.getcwd()
obj = analyzer.portable_analyzer(current_directory)

#colorword = 'Green'
checkerdata  = readXYZ_radiant(path, colorword +' Pos ' + angleword)

#checkerdata  = readXYZ_radiant(path, 'Green Pos ' + angleword)
checkerimage, r = convert2img(checkerdata[0][1])

negcheckerdata = readXYZ_radiant(path, colorword + ' Neg ' + angleword)
#negcheckerdata = readXYZ_radiant(path, 'Green Neg ' + angleword)

negcheckerimage, r = convert2img(negcheckerdata[0][1])

#colorword = 'Blue'

cleardata = readXYZ_radiant(path, colorword + ' Solid ' + angleword)
clearimage, brightratio = convert2img(cleardata[0][1])
clearimage_X, r_X = convert2img(cleardata[0][0])
clearimage_Z, r_Z = convert2img(cleardata[0][2])

# load crosshairs images and find crosshair coordinates

crosstemplate = cv2.imread(crosshairtemplatepath,cv2.IMREAD_GRAYSCALE)
crossdata = readXYZ_radiant(path,  'Green Cross ' + angleword)
crossimage,r = convert2img(crossdata[0][1])
crossimage = image_utils.rotate_image(crossimage, rotate)
crosshair_coordinates = image_utils.get_crosshairs(crossimage, crosstemplate, showcross = False)
center_m_dut = np.mean(crosshair_coordinates[:,0]) + center_m_dut_offset    #use crosshair coordinates to find the center of checkboard pattern
center_n_dut = np.mean(crosshair_coordinates[:,1]) + center_n_dut_offset


#rotate these images by 90 deg
checkerimage = image_utils.rotate_image(checkerimage, rotate)
negcheckerimage = image_utils.rotate_image(negcheckerimage, rotate)
clearimage = image_utils.rotate_image(clearimage, rotate)
clearimage_X = image_utils.rotate_image(clearimage_X, rotate)
clearimage_Z = image_utils.rotate_image(clearimage_Z, rotate)


#find coordinates of crosshairs




#checkerimage,K_mtx = image_utils.undistort(checkerimage, dist_coef)
#negcheckerimage,K_mtx = image_utils.undistort(negcheckerimage, dist_coef)

# checkerimage = image_utils.rotate_image(checkerimage,-3)
# negcheckerimage = image_utils.rotate_image(negcheckerimage,-3)


# checkerimage = cv2.flip(checkerimage,0)
# negcheckerimage = cv2.flip(negcheckerimage,0)


obj.load_clear(clearimage)

obj.load_checker_negchecker(checkerimage,negcheckerimage)
#obj.meancontrast(18, 18, squaresize_m = 37.5, squaresize_n = 38,  center_m = 781.8, center_n = 680.5)


# analyzer.save_to_csv(obj.contrastcsv, "Ucalibrated_contrastmap.csv", mode = 'w')
# meandata = [
#     ["Mean contrast",obj.meancontrast],
#     ["Harmonic mean contrast",obj.hmeancontrast]
#     ]
# analyzer.save_to_csv(meandata, "Ucalibrated_contrastmap.csv", mode = 'a')



#Getting SLB contrast and calibrate the contrast

slbcheckerdata  = readXYZ_radiant(slbpath, colorword + ' Pos')
slbcheckerimage, r = convert2img(slbcheckerdata[0][1])
slbnegcheckerdata = readXYZ_radiant(slbpath, colorword + ' Neg')
slbnegcheckerimage, r = convert2img(slbnegcheckerdata[0][1])

slbcleardata = readXYZ_radiant(slbpath, colorword + ' Solid')
slbclearimage, SLB_brightnessratio = convert2img(slbcleardata[0][1])
#slbcheckerimage,K_mtx = image_utils.undistort(slbcheckerimage, dist_coef)
#slbnegcheckerimage,K_mtx = image_utils.undistort(slbnegcheckerimage, dist_coef)

slbcheckerimage = image_utils.rotate_image(slbcheckerimage, rotate_slb)
slbnegcheckerimage = image_utils.rotate_image(slbnegcheckerimage, rotate_slb)
slbclearimage = image_utils.rotate_image(slbclearimage, rotate_slb)

#slbcheckerimage = cv2.flip(slbcheckerimage,0)
#slbnegcheckerimage = cv2.flip(slbnegcheckerimage,0)



obj.load_SLB(slbcheckerimage,slbnegcheckerimage,slbclearimage)
#obj.SLB_contrast(18, 18, squaresize_m = 38, squaresize_n = 38,  center_m = 828, center_n = 696)
#obj.meancontrast(18, 18, squaresize_m = 37.5, squaresize_n = 38,  center_m = 781.8, center_n = 680.5)
#obj.SLB_contrast(17, 17, squaresize_m = 37.5, squaresize_n = 37.5,  center_m = 794, center_n = 497)
#obj.SLB_contrast(16, 16, squaresize_m = 34, squaresize_n = 33,  center_m = 880, center_n = 625)
#obj.SLB_contrast(17, 17, squaresize_m = 37.5, squaresize_n = 37.5,  center_m = 782, center_n = 480)  #+5deg projector tilt
obj.SLB_contrast(17, 17, squaresize_m = squaresize_m_slb, squaresize_n = squaresize_n_slb,  center_m = center_m_slb, center_n = center_n_slb)
#obj.meancontrast(17, 17, squaresize_m = 37.3, squaresize_n = 37.5,  center_m = 868, center_n = 648)
obj.meancontrast(17, 17, squaresize_m = squaresize_m_dut, squaresize_n = squaresize_n_dut, saddlepoint_size = saddlepoint_size, detectable_saddlepoint = detectable_saddlepoint ,  center_m = center_m_dut, center_n = center_n_dut)
#obj.efficiency(FOVx = 16, FOVy = 8, pixperdeg = 18.75,exposure_SLB = 1, exposure_DUT = 1,center_m = None, center_n = None, ND_A_path = None, ND_B_path = None)

obj.contrastcalibration()

#obj.Radiant_efficiency(cleardata[0][1], slbcleardata[0][1], FOVx = 34, FOVy = 34, pixperdeg = 18.75, exposure_SLB = 1, exposure_DUT = 1,dut_center_m = 633, dut_center_n = 563,slb_center_m = 782,slb_center_n = 480, ND_A_path = None, ND_B_path = None)  #SLB +5deg tilt
brightness_DUT, brightness_SLB, uniformity_dut, uniformity_slb = obj.Radiant_efficiency(clearimage, slbclearimage, tilt_angle, FOVx = FOVx, FOVy = FOVy, pixperdeg = pixperdeg, exposure_SLB = 1, exposure_DUT = 1,dut_center_m = center_m_dut, dut_center_n = center_n_dut ,slb_center_m = center_m_slb, slb_center_n = center_n_slb, ND_A_path = None, ND_B_path = None)
brightness_DUT = brightness_DUT/brightratio
brightness_SLB = brightness_SLB/SLB_brightnessratio
obj.Rad_eff = obj.Rad_eff * SLB_brightnessratio/brightratio
obj.ZoneA_efficiency = obj.ZoneA_efficiency * SLB_brightnessratio/brightratio

obj.Get_FOV(dut_center_m = center_m_dut, dut_center_n = center_n_dut ,slb_center_m = center_m_slb, slb_center_n = center_n_slb)

#save cropped clear pattern to csv file

DUT_Y_crop = [round(center_n_dut - FOVy*pixperdeg/2), round(center_n_dut + FOVy*pixperdeg/2)]
DUT_X_crop = [round(center_m_dut - FOVx*pixperdeg/2), round(center_m_dut + FOVx*pixperdeg/2)]
X_crop = clearimage_X[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]]
Y_crop = clearimage[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]]
Z_crop = clearimage_Z[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]]

X_crop = X_crop/r_X
Y_crop = Y_crop/brightratio
Z_crop = Z_crop/r_Z


#Create the filenames of X_crop, Y_crop and Z_crop to match Colin's format
if pupil < 6:
    Region = 'reg0'
else:
    Region = 'reg1'

XCrop_name = 'X_' + colorword.lower() + '_' + Region + '_' + pupilnumber + '_' + Side + '_' + str(aperture_size) + 'mm_ET.csv'
YCrop_name = 'Y_' + colorword.lower() + '_' + Region + '_' + pupilnumber + '_' + Side + '_' + str(aperture_size) + 'mm_ET.csv'
ZCrop_name = 'Z_' + colorword.lower() + '_' + Region + '_' + pupilnumber + '_' + Side + '_' + str(aperture_size) + 'mm_ET.csv'

if colorword != 'White':
    analyzer.save_to_csv(X_crop, dirtory + '\\' + datafilename + '\\' + XCrop_name, mode = 'w')
    analyzer.save_to_csv(Y_crop, dirtory + '\\' + datafilename + '\\' + YCrop_name, mode = 'w')
    analyzer.save_to_csv(Z_crop, dirtory + '\\' + datafilename + '\\' + ZCrop_name, mode = 'w')

print("FOV_H_center is {:.5f}".format(obj.FOV_dut_H_center))
print("FOV_V_center is {:.5f}".format(obj.FOV_dut_V_center))
print("SLBFOV_H_center is {:.5f}".format(obj.FOV_slb_H))
print("SLBFOV_V_center is {:.5f}".format(obj.FOV_slb_V))

#output data and config files

config_data =  [
     ["aperture_size(mm)",aperture_size],
     ["Eyebox_size(mm^2)",eyebox_size],
     ['center_m_dut',center_m_dut],
     ['center_n_dut',center_n_dut],
     ['center_m_slb',center_m_slb],
     ['center_n_slb',center_n_slb],
     ['rotate_dut(deg)',rotate],
     ['rotate_slb(deg)',rotate_slb],
     ['squaresize_m_dut',squaresize_m_dut],
     ['squaresize_n_dut',squaresize_n_dut],
     ['squaresize_m_slb',squaresize_m_slb],
     ['squaresize_n_slb',squaresize_n_slb],
     ]
analyzer.save_to_csv(config_data, "Config.csv", mode = 'w')

print("SLB Checkerboard mean contrast is {:.5f}".format(obj.slb_meancontrast))
print("SLB Checkerboard harmonic mean contrast is {:.5f}".format(obj.slb_hmeancontrast))
print("Ucalibrated Checkerboard mean contrast is {:.5f}".format(obj.meancontrast))
print("Ucalibrated Checkerboard harmonic mean contrast is {:.5f}".format(obj.hmeancontrast))
print("Calibrated Checkerboard mean contrast is {:.5f}".format(obj.calibrated_meancontrast))
print("Calibrated Checkerboard harmonic mean contrast is {:.5f}".format(obj.calibrated_hmeancontrast))
print("Efficiency is {:.5f}".format(obj.Rad_eff))
print("Eyebox Efficiency is {:.5f}".format(obj.Rad_eff * eyebox_size/(np.pi*aperture_size**2/4)))  

cv2.imwrite('checker.png',obj.checker)
cv2.imwrite('negchecker.png',obj.negchecker)
cv2.imwrite('slbchecker.png',obj.slbchecker)
cv2.imwrite('slbnegchecker.png',obj.slbnegchecker)
cv2.imwrite('slbclear.png',obj.slbclear_pattern)
cv2.imwrite('clear.png',obj.clear_pattern)


analyzer.save_to_csv(obj.slb_contrastcsv, "SLB_contrastmap.csv", mode = 'w')
analyzer.save_to_csv(obj.contrastcsv, "Ucalibrated_contrastmap.csv", mode = 'w')
analyzer.save_to_csv(obj.calibrated_contrastcsv, "Calibrated_contrastmap.csv", mode = 'w')


meandata = [
    ["Mean contrast",obj.meancontrast],
    ["Harmonic mean contrast",obj.hmeancontrast]
    ]
analyzer.save_to_csv(meandata, "Ucalibrated_contrastmap.csv", mode = 'a')

slbmeandata = [
    ["SLB_Meancontrast",obj.slb_meancontrast],
    ["SLB_Harmonic mean contrast",obj.slb_hmeancontrast]
    ]
analyzer.save_to_csv(slbmeandata, "SLB_contrastmap.csv", mode = 'a')

calibrated_meandata = [
    ["Calibrated_Meancontrast",obj.calibrated_meancontrast],
    ["ZoneA_Calibrated_Meancontrast", obj.ZoneA_meancontrast],
    ["Calibrated_Harmonic mean contrast",obj.calibrated_hmeancontrast]
    ]
analyzer.save_to_csv(calibrated_meandata, "Calibrated_contrastmap.csv", mode = 'a')

efficiency_data = [
     ["Efficiency",obj.Rad_eff],
     ["ZoneA_efficiency", obj.ZoneA_efficiency],
     ["Eyebox Efficiency",obj.Rad_eff * eyebox_size/(np.pi*aperture_size**2/4)]
     ]
analyzer.save_to_csv(efficiency_data, "Efficiency.csv", mode = 'w')

SLB_result_data  = [
    ["DUT_brightness(nit)",brightness_DUT],
    ["SLB_brightness(nit)",brightness_SLB],
    ["DUT_angular_uniformity",uniformity_dut],
    ["SLB_angular_uniformity",uniformity_slb],
    ["ZoneA_angular_uniformity", obj.ZoneA_uniformity]
]
analyzer.save_to_csv(SLB_result_data, "SLB_result.csv", mode = 'w')