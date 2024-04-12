# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:57:15 2023

Data analyzer for portable miniWG tester

@author: Zhida Xu
"""

import os
import fnmatch
import numpy as np
import cv2
from matplotlib import pyplot
import glob
import pandas as pd
import csv
#import MTF_crosshair
import json
import re
from scipy.ndimage import median_filter
import pdb #for debug

ix = -1
iy = -1

crosscenters = []

img = []
windowname = 'hut'

class portable_analyzer():
    def __init__(self,path = None, debug = False) -> None:
        self.debug = debug
        #self.gamma = 'g'
        self.gamma = 'b'
        if path: 
            self.path = path
            if os.path.isdir(path):
                os.chdir(path)
    
    def load_clear(self,clearpath):
        self.clear = clearpath
        #self.clear = self.clear - self.darkimage
        if type(self.gamma) == pd.pandas.core.frame.DataFrame:
            self.clear = self.gamma_correction(self.clear)
    
    def load_NDfilter(self, ND_A_path, ND_B_path):
        self.ND_A = cv2.imread(ND_A_path,cv2.IMREAD_GRAYSCALE) - self.darkimage
        self.ND_B = cv2.imread(ND_B_path,cv2.IMREAD_GRAYSCALE) - self.darkimage
        if type(self.gamma) == pd.pandas.core.frame.DataFrame:
            self.ND_A = self.gamma_correction(self.ND_A)
            self.ND_B = self.gamma_correction(self.ND_B)
    
    
    #need to flip the SLB images
    def load_SLB(self, slbcheckerpath,slbnegcheckerpath, slbclearpath):  
        self.slbchecker = slbcheckerpath
        self.slbnegchecker = slbnegcheckerpath
        self.slbclear = slbclearpath
        if type(self.gamma) == pd.pandas.core.frame.DataFrame:
            self.slbchecker = self.gamma_correction(self.slbchecker)
            self.slbnegchecker = self.gamma_correction(self.slbnegchecker)
            self.slbclear = self.gamma_correction(self.slbclear)
        
            
    def load_dark(self,darkpath):
        #darkimagenames = glob.glob("*.bmp")
        self.darkimage = cv2.imread(darkpath,cv2.IMREAD_GRAYSCALE)
        self.width = self.darkimage.shape[1]
        self.height = self.darkimage.shape[0]
        
    def load_checker_negchecker(self,checkerpath, negcheckerpath):
        self.checker = checkerpath
        self.negchecker = negcheckerpath
        if type(self.gamma) == pd.pandas.core.frame.DataFrame:
            self.checker = self.gamma_correction(self.checker)
            self.negchecker = self.gamma_correction(self.negchecker)
        
    def load_ghost(self,ghostpath):
        self.ghost = ghostpath
        if type(self.gamma) == pd.pandas.core.frame.DataFrame:      
            self.ghost = self.gamma_correction(self.ghost)
        self.displayghost = self.ghost.copy()
        
    def load_cross(self,crosspath):
        self.cross = crosspath
        if type(self.gamma) == pd.pandas.core.frame.DataFrame:
            self.cross = self.gamma_correction(self.cross)
        self.displaycross = self.cross.copy()
    
    def load_gamma(self,gamma_path):
        self.gamma = 128 * pd.read_excel(gamma_path)
    
    def gamma_correction(self,image,AOI_portion = 0.5):
        #gamma = 255* pd.read_excel(gamma_path)
        [h,w] = image.shape 
        
        #crop_image = image[ int((h-h*AOI_portion)/2):int((h+h*AOI_portion)/2), int((w-w*AOI_portion)/2):int((w + w*AOI_portion)/2)]
        corrected_image = image.copy()
        H = int((h+h*AOI_portion)/2) - int((h-h*AOI_portion)/2)
        W = int((w + w*AOI_portion)/2) - int((w-w*AOI_portion)/2)
        #[H,W] = crop_image.shape
        #corrected_image = np.zeros((H,W))
        for i in range(H):
            for j in range(W):
                corrected_image[int((h-h*AOI_portion)/2) + i,int((w-w*AOI_portion)/2) + j] = np.uint8(round(np.interp(corrected_image[int((h-h*AOI_portion)/2) + i,int((w-w*AOI_portion)/2) + j], self.gamma.iloc[:, 1], self.gamma.iloc[:, 0])))
        return corrected_image
        
    def efficiency(self, FOVx = 16, FOVy = 8, pixperdeg = 67.566,exposure_SLB = 1, exposure_DUT = 1,center_m = None, center_n = None, ND_A_path = None, ND_B_path = None):       #Also return the FOV in H, V                       
        [H,W] = self.clear.shape
        if center_m == None or center_n == None:
            center_m = W/2
            center_n = H/2
        Y_crop = [round(center_n - FOVy*pixperdeg/2), round(center_n + FOVy*pixperdeg/2)]
        X_crop = [round(center_m - FOVx*pixperdeg/2), round(center_m + FOVx*pixperdeg/2)]
        
        if ND_A_path == None or ND_B_path == None:
            transmission_ratio_m = 0.06
        else:
            self.load_NDfilter(ND_A_path,ND_B_path)
            if np.mean(self.ND_A) > np.mean(self.ND_B):
                transmission_ratio_m = np.array(self.ND_B[Y_crop[0]:Y_crop[1],X_crop[0]:X_crop[1]]) / np.array(self.ND_A[Y_crop[0]:Y_crop[1],X_crop[0]:X_crop[1]])
            else:
                transmission_ratio_m = np.array(self.ND_A[Y_crop[0]:Y_crop[1],X_crop[0]:X_crop[1]]) / np.array(self.ND_B)
        
        self.eff_map = (np.array(self.clear[Y_crop[0]:Y_crop[1],X_crop[0]:X_crop[1]]) * transmission_ratio_m * exposure_SLB) / (np.array(self.slbclear[Y_crop[0]:Y_crop[1],X_crop[0]:X_crop[1]]) * exposure_DUT)
        self.eff = np.mean(self.eff_map)
        
        self.slbclear_pattern = self.slbclear.copy()
        self.clear_pattern = self.clear.copy()
        
        cv2.rectangle(self.slbclear_pattern, (X_crop[0],Y_crop[0]),(X_crop[1],Y_crop[1]), (30000, 30000, 30000), 2)
        cv2.rectangle(self.clear_pattern, ( X_crop[0],Y_crop[0]),(X_crop[1],Y_crop[1]), (30000, 30000, 30000), 2)
        
        
    
    def Radiant_efficiency(self, DUT_cleardata, SLB_cleardata, Duttiltangle, FOVx = 16, FOVy = 8, pixperdeg = 18.75, exposure_SLB = 1, exposure_DUT = 1,dut_center_m = None, dut_center_n = None,slb_center_m = None,slb_center_n = None, ND_A_path = None, ND_B_path = None):                            
        [H,W] = self.clear.shape
        if dut_center_m == None or dut_center_n == None:
            dut_center_m = W/2
            dut_center_n = H/2
        if slb_center_m == None or slb_center_n == None:
            slb_center_m = W/2
            slb_center_n = H/2
            
        DUT_Y_crop = [round(dut_center_n - FOVy*pixperdeg/2), round(dut_center_n + FOVy*pixperdeg/2)]
        DUT_X_crop = [round(dut_center_m - FOVx*pixperdeg/2), round(dut_center_m + FOVx*pixperdeg/2)]
        SLB_Y_crop = [round(slb_center_n - FOVy*pixperdeg/2), round(slb_center_n + FOVy*pixperdeg/2)]
        SLB_X_crop = [round(slb_center_m - FOVx*pixperdeg/2), round(slb_center_m + FOVx*pixperdeg/2)]
        
        
        if ND_A_path == None or ND_B_path == None:
            transmission_ratio_m = 1
        else:
            self.load_NDfilter(ND_A_path,ND_B_path)
            if np.mean(self.ND_A) > np.mean(self.ND_B):
                transmission_ratio_m = np.array(self.ND_B[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]]) / np.array(self.ND_A[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]])
            else:
                transmission_ratio_m = np.array(self.ND_A[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]]) / np.array(self.ND_B[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]])
        
        #self.Rad_eff_map = (np.array(DUT_cleardata[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]]) * transmission_ratio_m * exposure_SLB) / (np.array(SLB_cleardata[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]]) * exposure_DUT)
        
        #self.Rad_eff = np.mean(self.Rad_eff_map)
        
        #self.Rad_eff = np.mean((np.array(DUT_cleardata[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]])) * transmission_ratio_m * exposure_SLB) / (np.mean(np.array(SLB_cleardata[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]])) * exposure_DUT)
        self.cropclear_DUT = np.array(DUT_cleardata[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]])
        self.cropclear_SLB = np.array(SLB_cleardata[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]])
        
        #brightness_DUT = np.mean(np.array(DUT_cleardata[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]]))
        #brightness_SLB = np.mean(np.array(SLB_cleardata[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]]))
        brightness_DUT = np.mean(self.cropclear_DUT)
        brightness_SLB = np.mean(self.cropclear_SLB)
        
        
        self.Rad_eff = brightness_DUT/brightness_SLB
        
        #Calculate angular uniformity of 99/1 prtl inside efficiency function
        filter_size = (round(pixperdeg/2 + 1), round(pixperdeg/2 + 1))
        #clear_crop_blur = median_filter(np.array(DUT_cleardata[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]]),size=filter_size)
        #SLB_crop_blur = median_filter(np.array(SLB_cleardata[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]]),size=filter_size)

        clear_crop_blur = median_filter(self.cropclear_DUT,size=filter_size)
        SLB_crop_blur = median_filter(self.cropclear_DUT,size=filter_size)        
        
        
        [clear_min, clear_max] = np.percentile(clear_crop_blur,[5,95])
        [slb_min, slb_max] = np.percentile(SLB_crop_blur,[5,95])
        
        uniformity_dut = clear_max/clear_min
        uniformity_slb = slb_max / slb_min
        
        self.slbclear_pattern = self.slbclear.copy()
        self.clear_pattern = self.clear.copy()
        
        
        cv2.rectangle(self.slbclear_pattern, (SLB_X_crop[0],SLB_Y_crop[0]),(SLB_X_crop[1],SLB_Y_crop[1]), (30000, 30000, 30000), 2)
        cv2.rectangle(self.clear_pattern, (DUT_X_crop[0],DUT_Y_crop[0]),(DUT_X_crop[1],DUT_Y_crop[1]), (30000, 30000, 30000), 2)
        
        self.get_ZoneA_metrics(DUT_X_crop, DUT_Y_crop, SLB_X_crop, SLB_Y_crop, Duttiltangle)
        
        return brightness_DUT, brightness_SLB, uniformity_dut, uniformity_slb


    def Get_FOV(self, threshold = 0.1, inset_H = 1, inset_V = 3,FOVx = 34, FOVy = 34, pixperdeg = 18.75, dut_center_m = None, dut_center_n = None,slb_center_m = None,slb_center_n = None): #inset in degs
        [H,W] = self.clear.shape
        if dut_center_m == None or dut_center_n == None:
            dut_center_m = W/2
            dut_center_n = H/2
        if slb_center_m == None or slb_center_n == None:
            slb_center_m = W/2
            slb_center_n = H/2
            
        DUT_Y_crop = [round(dut_center_n - FOVy*pixperdeg/2), round(dut_center_n + FOVy*pixperdeg/2)]
        DUT_X_crop = [round(dut_center_m - FOVx*pixperdeg/2), round(dut_center_m + FOVx*pixperdeg/2)]
        SLB_Y_crop = [round(slb_center_n - FOVy*pixperdeg/2), round(slb_center_n + FOVy*pixperdeg/2)]
        SLB_X_crop = [round(slb_center_m - FOVx*pixperdeg/2), round(slb_center_m + FOVx*pixperdeg/2)]
        
        
        dutFOV = self.clear[DUT_Y_crop[0]:DUT_Y_crop[1],DUT_X_crop[0]:DUT_X_crop[1]]
        slbFOV = self.slbclear[SLB_Y_crop[0]:SLB_Y_crop[1],SLB_X_crop[0]:SLB_X_crop[1]]
        dut_mean = np.mean(np.array(dutFOV))
        slb_mean = np.mean(np.array(slbFOV))
        self.FOV_dut_H_center = sum(dutFOV[round((DUT_Y_crop[1] - DUT_Y_crop[0])/2),:] > threshold * dut_mean) / pixperdeg
        self.FOV_slb_H_center = sum(slbFOV[round((SLB_Y_crop[1] - SLB_Y_crop[0])/2),:] > threshold * slb_mean) / pixperdeg
        self.FOV_dut_V_center = sum(dutFOV[:,round((DUT_X_crop[1] - DUT_X_crop[0])/2)] > threshold * dut_mean) / pixperdeg
        self.FOV_slb_V_center = sum(slbFOV[:,round((DUT_X_crop[1] - DUT_Y_crop[0])/2)] > threshold * slb_mean) / pixperdeg
        
        self.FOV_dut_H = np.min([sum(dutFOV[round(inset_V*pixperdeg),:]> threshold * dut_mean) ,sum(dutFOV[-1-round(inset_V*pixperdeg),:]> threshold * dut_mean) ]) / pixperdeg
        self.FOV_slb_H = np.min([sum(slbFOV[round(inset_V*pixperdeg),:]> threshold * slb_mean) ,sum(slbFOV[-1-round(inset_V*pixperdeg),:]> threshold * slb_mean) ]) / pixperdeg
        self.FOV_dut_V = np.min([sum(dutFOV[:,round(inset_H*pixperdeg)]> threshold * dut_mean) ,sum(dutFOV[:,-1-round(inset_H*pixperdeg)]> threshold * dut_mean) ]) / pixperdeg
        self.FOV_slb_V = np.min([sum(slbFOV[:,round(inset_H*pixperdeg)]> threshold * slb_mean) ,sum(slbFOV[:,-1-round(inset_H*pixperdeg)]> threshold * slb_mean) ]) / pixperdeg
        
        

    def meancontrast(self,checker_m,checker_n,squaresize_m = 127, squaresize_n = 127,ROI_portion = 0.5, saddlepoint_size = (16,16), detectable_saddlepoint = 16  ,center_m = None,center_n = None):    #checker_m is number of columns, #checker_n is number of rows
        if center_m == None:
            center_m = self.width/2
        if center_n == None:
            center_n = self.height/2
        
        #center_m = self.width - center_m
        #center_n = self.height - center_n
        ROIsize_m = round(squaresize_m * ROI_portion)
        ROIsize_n = round(squaresize_n * ROI_portion)
        ROIcenters = [[[0]*2 for i in range(checker_m)] for j in range(checker_n)]   #j is index of rows, j is index of columns
        img = image_enhancement_with_bg(self.checker,self.clear)
        center_list = get_checkerboard_center(img, saddlepoint_size, (saddlepoint_size[0], detectable_saddlepoint), saddlepoint_size[1] - detectable_saddlepoint, degree =3)  #degree is extropolation order
        self.meancontrast = 0
        self.hmeancontrast = 0
        self.contrastcsv =  [[[-1] for i in range(checker_m)] for j in range(checker_n)]  
        self.Contrastmap =  [[[0]*3 for i in range(checker_m)] for j in range(checker_n)]   # order is checker brigthness, negchecker brightness and contrast for ROI of each checker
        for i in range(checker_m):
            for j in range(checker_n):
                #ROIcenters[j][i] = [j * squaresize_n + center_n - (checker_n-1)*squaresize_n/2, i * squaresize_m + center_m - (checker_m-1)*squaresize_m/2] 
                #ROIcenters[j][i] = swap(center_list[j*checker_m+i][:])
                cj = checker_m-1-j #index y flip
                ci = checker_n-1-i #index x flip
                ROIcenters[j][i] = (center_list[cj*checker_m+ci][1],center_list[cj*checker_m+ci][0])

                self.Contrastmap[j][i][0] = np.mean(self.checker[int(ROIcenters[j][i][0] - round(ROIsize_n/2))-1: int(ROIcenters[j][i][0] + round(ROIsize_n/2)) +1 , int(ROIcenters[j][i][1]-round(ROIsize_m/2)) -1 : int(ROIcenters[j][i][1] + round(ROIsize_m/2)) + 1 ])     #calculate the mean brigthenss for this checker
                self.Contrastmap[j][i][1] = np.mean(self.negchecker[int(ROIcenters[j][i][0] - round(ROIsize_n/2))-1: int(ROIcenters[j][i][0] + round(ROIsize_n/2)) +1 , int(ROIcenters[j][i][1]-round(ROIsize_m/2)) -1 : int(ROIcenters[j][i][1] + round(ROIsize_m/2)) + 1 ])
                if  self.Contrastmap[j][i][0] >  self.Contrastmap[j][i][1]:
                    self.contrastcsv[j][i]  = self.Contrastmap[j][i][2] =  self.Contrastmap[j][i][0]/ self.Contrastmap[j][i][1]                    
                else:
                    self.Contrastmap[j][i][2] =  self.Contrastmap[j][i][1]/ self.Contrastmap[j][i][0]
                    self.contrastcsv[j][i] = self.Contrastmap[j][i][1]/ self.Contrastmap[j][i][0]
                self.meancontrast = self.meancontrast + self.Contrastmap[j][i][2]
                self.hmeancontrast = self.hmeancontrast + 1.0/self.Contrastmap[j][i][2]
                cv2.rectangle(self.checker, (int(ROIcenters[j][i][1]-round(ROIsize_m/2)),int(ROIcenters[j][i][0] - round(ROIsize_n/2))),  (int(ROIcenters[j][i][1] + round(ROIsize_m/2)),int(ROIcenters[j][i][0] + round(ROIsize_n/2))) , (30000, 30000, 30000), 2)
                cv2.rectangle(self.negchecker, (int(ROIcenters[j][i][1]-round(ROIsize_m/2)),int(ROIcenters[j][i][0] - round(ROIsize_n/2))),  (int(ROIcenters[j][i][1] + round(ROIsize_m/2)),int(ROIcenters[j][i][0] + round(ROIsize_n/2))) , (30000, 30000, 30000), 2)                
        self.meancontrast = self.meancontrast/(checker_m * checker_n)
        self.hmeancontrast = checker_m * checker_n/self.hmeancontrast          

        # Need to add SLB checker and negcheck contrast map and calibrate the DUT checker and negchecker to SLB checker contrast

    def SLB_contrast(self,checker_m,checker_n,squaresize_m = 127, squaresize_n = 127,ROI_portion = 0.5,center_m = None,center_n = None):    #checker_m is number of columns, #checker_n is number of rows
        if center_m == None:
            center_m = self.width/2
        if center_n == None:
            center_n = self.height/2
        
        #center_m = self.width - center_m
        #center_n = self.height - center_n
        ROIsize_m = round(squaresize_m * ROI_portion)
        ROIsize_n = round(squaresize_n * ROI_portion)
        ROIcenters = [[[0]*2 for i in range(checker_m)] for j in range(checker_n)]   #j is index of rows, j is index of columns
        img = image_enhancement_with_bg(self.slbchecker,self.slbclear)
        #pdb.set_trace()  #breakpoint
        center_list = get_checkerboard_center(img, (checker_m - 1,checker_n - 1), (checker_m - 1,checker_n -1), 0, degree = 0)  #degree is extropolation order
        self.slb_meancontrast = 0
        self.slb_hmeancontrast = 0
        self.slb_contrastcsv =  [[[-1] for i in range(checker_m)] for j in range(checker_n)]  
        self.SLBContrastmap =  [[[0]*3 for i in range(checker_m)] for j in range(checker_n)]   # order is checker brigthness, negchecker brightness and contrast for ROI of each checker
        for i in range(checker_m):
            for j in range(checker_n):
                #ROIcenters[j][i] = [j * squaresize_n + center_n - (checker_n-1)*squaresize_n/2, i * squaresize_m + center_m - (checker_m-1)*squaresize_m/2] 
                #ROIcenters[j][i] = [j * squaresize_n + center_n - (checker_n-1)*squaresize_n/2, i * squaresize_m + center_m - (checker_m-1)*squaresize_m/2] 
                ROIcenters[j][i] = (center_list[j*checker_m+i][1],center_list[j*checker_m+i][0]) #swap(center_list[j*checker_m+i][:])
                self.SLBContrastmap[j][i][0] = np.mean(self.slbchecker[int(ROIcenters[j][i][0] - round(ROIsize_n/2))-1: int(ROIcenters[j][i][0] + round(ROIsize_n/2)) +1 , int(ROIcenters[j][i][1]-round(ROIsize_m/2)) -1 : int(ROIcenters[j][i][1] + round(ROIsize_m/2)) + 1 ])     #calculate the mean brigthenss for this checker
                self.SLBContrastmap[j][i][1] = np.mean(self.slbnegchecker[int(ROIcenters[j][i][0] - round(ROIsize_n/2))-1: int(ROIcenters[j][i][0] + round(ROIsize_n/2)) +1 , int(ROIcenters[j][i][1]-round(ROIsize_m/2)) -1 : int(ROIcenters[j][i][1] + round(ROIsize_m/2)) + 1 ])
                if  self.SLBContrastmap[j][i][0] >  self.SLBContrastmap[j][i][1]:
                    self.slb_contrastcsv[j][i]  = self.SLBContrastmap[j][i][2] =  self.SLBContrastmap[j][i][0]/ self.SLBContrastmap[j][i][1]                    
                else:
                    self.SLBContrastmap[j][i][2] =  self.SLBContrastmap[j][i][1]/ self.SLBContrastmap[j][i][0]
                    self.slb_contrastcsv[j][i] = self.SLBContrastmap[j][i][1]/ self.SLBContrastmap[j][i][0]
                self.slb_meancontrast = self.slb_meancontrast + self.SLBContrastmap[j][i][2]
                self.slb_hmeancontrast = self.slb_hmeancontrast + 1.0/self.SLBContrastmap[j][i][2]
                cv2.rectangle(self.slbchecker, (int(ROIcenters[j][i][1]-round(ROIsize_m/2)),int(ROIcenters[j][i][0] - round(ROIsize_n/2))),  (int(ROIcenters[j][i][1] + round(ROIsize_m/2)),int(ROIcenters[j][i][0] + round(ROIsize_n/2))) , (30000, 30000, 30000), 2)
        self.slb_meancontrast = self.slb_meancontrast/(checker_m * checker_n)
        self.slb_hmeancontrast = checker_m * checker_n/self.slb_hmeancontrast  
    
    def contrastcalibration(self):
        checker_n = len(self.contrastcsv)
        checker_m = len(self.contrastcsv[0])
        self.calibrated_contrastcsv =  [[[-1] for i in range(checker_m)] for j in range(checker_n)]
        self.calibrated_meancontrast = 0
        self.calibrated_hmeancontrast = 0
        for idx, e in np.ndenumerate(self.contrastcsv):
            self.calibrated_contrastcsv[idx[0]][idx[1]] = 1.0/(1.0/self.contrastcsv[idx[0]][idx[1]] - 1.0/self.slb_contrastcsv[idx[0]][idx[1]])
            self.calibrated_meancontrast = self.calibrated_meancontrast + self.calibrated_contrastcsv[idx[0]][idx[1]]
            self.calibrated_hmeancontrast = self.calibrated_hmeancontrast + 1.0/self.calibrated_contrastcsv[idx[0]][idx[1]]
        self.calibrated_meancontrast = self.calibrated_meancontrast/(checker_m * checker_n)
        self.calibrated_hmeancontrast = checker_m * checker_n/self.calibrated_hmeancontrast
        
        
    
    
    def Ghost(self):  #repeated chosen ROI for ghost and original image
        [H,W] = self.ghost.shape
        self.points = [(-1,-1)]*4
        
        #ghostflag = False
        self.n = 0   
        self.ghostdata = pd.DataFrame(columns=["source_x","source_y","ghost_x","ghost_y", "source_brightness", "ghost_brigthness","ghost_strength"])
        
        self.Windowname = "first pick topleft corner then pick bottom right corner, must be in order, click any key to quit"
        
        cv2.namedWindow(self.Windowname,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(self.Windowname,self.displayghost)
        cv2.setMouseCallback(self.Windowname, self.draw_rectangle)
        
        # while True:
        #     cv2.imshow("Ghost picking, click q to quit",displayghost)
        #     key = cv2.waitKey(1) & 0xFF        
            
            
        #     if key == ord('q') or key == 27:
        #         break
        
        cv2.waitKey(0)        
        cv2.destroyAllWindows()
        #cv2.imwrite('Ghost.png',self.displayghost)
        
            
        
    def draw_rectangle(self,event, x, y, flags, param):
        global ghostflag, top_left_clicked, bottom_right_clicked
        
        #print((x,y))
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.n % 4 == 0:   #topleft corner of source image
                self.points[0] = (x,y)
                cv2.circle(self.displayghost, self.points[0], radius= 2, color=(30000, 30000, 30000), thickness= -1)
            if self.n% 4 == 1: 
                self.points[1] = (x,y)
                cv2.rectangle(self.displayghost, self.points[0], self.points[1], (12000,12000,12000), 2)
                cv2.putText(self.displayghost, "Source# {:d}".format(self.n//4 + 1), self.points[1], cv2.FONT_HERSHEY_SIMPLEX, 1, (30000, 30000, 0), 2)
            if self.n% 4 == 2: 
                self.points[2] = (x,y)
                cv2.circle(self.displayghost, self.points[2], radius= 2, color=(30000, 30000, 30000), thickness= -1)
            if self.n% 4 == 3: 
                self.points[3] = (x,y) 
                cv2.rectangle(self.displayghost, self.points[2], self.points[3], (30000,30000,30000), 2)
                ghostbrightness = np.mean(self.ghost[self.points[2][1]:self.points[3][1],self.points[2][0]:self.points[3][0]])
                sourcebrightness = np.mean(self.ghost[self.points[0][1]:self.points[1][1],self.points[0][0]:self.points[1][0]])
                #self.ghostdata.append(pd.DataFrame([self.points[0][0],self.points[0][1],self.points[1][0],self.points[1][1],sourcebrightness,ghostbrightness,ghostbrightness/sourcebrightness]),ignore_index=True) 
                rowtoappend = {"source_x":self.points[0][0]
                               ,"source_y":self.points[0][1]
                                   ,"ghost_x":self.points[1][0]
                                       ,"ghost_y":self.points[1][1]
                                           ,"source_brightness":sourcebrightness
                                               ,"ghost_brigthness":ghostbrightness
                                                   ,"ghost_strength":ghostbrightness/sourcebrightness}
                
                self.ghostdata = self.ghostdata.append(rowtoappend, ignore_index=True)
                cv2.putText(self.displayghost, "Ghost# {:d},strengh: {:.4f}".format(self.n//4 + 1,ghostbrightness/sourcebrightness), self.points[3], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)                
                
            print("point {:d}: ({:d}, {:d})".format(self.n%4 + 1,x,y))
            #cv2.putText(self.displayghost, "What", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow(self.Windowname,self.displayghost)
            self.n = self.n + 1
        
     #def store_data(self):
    def angular_uniformity(self,FOVx, FOVy, pixperdeg = 67.566):  #FoVx, FoVy in deg
        [H,W] = self.clear.shape
        clearpattern = cv2.medianBlur(self.clear,round(pixperdeg/2 + 1))
        Y_crop = [round((H - FOVy*pixperdeg)/2) , round((H + FOVy*pixperdeg)/2)]
        X_crop = [round((W - FOVx*pixperdeg)/2) , round((W + FOVx*pixperdeg)/2)]
        croppattern = clearpattern[Y_crop[0]:Y_crop[1],X_crop[0]:X_crop[1]]
        [self.min, self.max] = np.percentile(croppattern,[1,99])
        self.uniformity = self.max/self.min
        return self.uniformity
    
    def get_ZoneA_metrics(self, DUT_X_crop, DUT_Y_crop, SLB_X_crop, SLB_Y_crop, duttiltangle = -5, slbtiltangle = 0, ZoneA_FOV = 23, pixperdeg = 18.75, checkerdeg = 2):
        [SLB_H, SLB_W] = self.cropclear_DUT.shape
        [DUT_H, DUT_W] = self.cropclear_SLB.shape
        dut_center_x = round(DUT_W/2 + duttiltangle * pixperdeg)
        dut_center_y = round(DUT_H/2)
        slb_center_x = round(SLB_W/2 + slbtiltangle * pixperdeg)
        slb_center_y = round(SLB_H/2)
        radius = ZoneA_FOV * pixperdeg/2
        ZoneA_clearDUT =  get_data_within_circle(self.cropclear_DUT, (dut_center_x, dut_center_y), radius)
        ZoneA_clearSLB =  get_data_within_circle(self.cropclear_SLB, (slb_center_x, slb_center_y), radius)
        
        self.ZoneA_efficiency = np.mean(ZoneA_clearDUT) / np.mean(ZoneA_clearSLB)
        #filter_size = (round(pixperdeg/2 + 1), round(pixperdeg/2 + 1))
        
        #ZoneA_clear_blur = median_filter(ZoneA_clearDUT ,size=filter_size)
        
        [clear_min, clear_max] = np.percentile(ZoneA_clearDUT,[5,95])
        self.ZoneA_uniformity = clear_max/clear_min
        
        calibratedcontrastmap = np.array(self.calibrated_contrastcsv)
        rawcontrastmap = np.array(self.contrastcsv)
        
        [C_H, C_W] = calibratedcontrastmap.shape #contrast map
        ZoneA_dut_contrast_center_x = round(C_W/2 + duttiltangle/checkerdeg) 
        ZoneA_dut_contrast_center_y = round(C_H/2)
        radius_contrast = 0.5* ZoneA_FOV/checkerdeg
        
        #pdb.set_trace()
        ZoneA_contrast_map = get_data_within_circle(calibratedcontrastmap, (ZoneA_dut_contrast_center_x, ZoneA_dut_contrast_center_y), radius_contrast)
        ZoneA_rawcontrast_map = get_data_within_circle(rawcontrastmap, (ZoneA_dut_contrast_center_x, ZoneA_dut_contrast_center_y), radius_contrast)
        
        self.ZoneA_meancontrast =  np.mean(ZoneA_contrast_map)
        self.ZoneA_meanrawcontrast = np.mean(ZoneA_rawcontrast_map)
        
        
        cv2.circle(self.clear_pattern, (DUT_X_crop[0] + dut_center_x, DUT_Y_crop[0] + dut_center_y), round(radius), (30000, 30000, 30000), 2)  #Draw zoneA on DUT clear pattern image
        cv2.circle(self.slbclear_pattern, (SLB_X_crop[0] + slb_center_x, SLB_Y_crop[0] + slb_center_y), round(radius), (30000, 30000, 30000), 2) #Draw ZoneA on SLB clear pattern image
        
        
        
    
    def caculate_crosshair_MTF(self, ROIwidth = 200, ROIheight = 100, ROIoffset = 200, pixperdeg = 67.566, freqs = [5,7.5,15,25], MTF_spreadsheet = 'crosshairs_MTFcurves.csv', MTF_info = 'MTF_info.csv'):
        global crosscenters
        ROI_labels = ['ROI-L','ROI-R','ROI-U','ROI-D']
        MTFcurves = pd.DataFrame()
        MTFinfo = pd.DataFrame()
        
        
        
        MTFinfo_1stcol = ['Crosscenter_X','Crosscenter_Y','Width','Height','Direction']
        
        for freq in freqs:
            MTFinfo_1stcol.append("MTF@{:.2f}lp/deg".format(freq))
            
        MTFinfo['Metric'] = MTFinfo_1stcol
        
        for idx,point in enumerate(crosscenters):
            cross_x = crosscenters[idx][0]
            cross_y = crosscenters[idx][1]
            
            df = [0]*4
            ROIs = [0]*4
            MTFs = [0]*4
            MTF_fs = [0]*4
            
            # ROI_L = self.displaycross[cross_y - int(ROIwidth/2)-1 : cross_y + int(ROIwidth/2), cross_x - ROIoffset-int(ROIheight/2) -1 : cross_x - ROIoffset + int(ROIheight/2)]   
            # ROI_R = self.displaycross[cross_y - int(ROIwidth/2)-1 : cross_y + int(ROIwidth/2), cross_x + ROIoffset-int(ROIheight/2) -1 : cross_x + ROIoffset + int(ROIheight/2)]
            # ROI_U = self.displaycross[cross_y - ROIoffset - int(ROIheight/2) -1 :  cross_y - ROIoffset + int(ROIheight/2), cross_x-int(ROIwidth/2) -1 : cross_x + int(ROIwidth/2)]
            # ROI_D = self.displaycross[cross_y + ROIoffset - int(ROIheight/2) -1 :  cross_y + ROIoffset + int(ROIheight/2), cross_x-int(ROIwidth/2) -1 : cross_x + int(ROIwidth/2)]
            
            ROIs[0] = self.displaycross[cross_y - int(ROIwidth/2)-1 : cross_y + int(ROIwidth/2), cross_x - ROIoffset-int(ROIheight/2) -1 : cross_x - ROIoffset + int(ROIheight/2)]   #ROI_L
            ROIs[1] = self.displaycross[cross_y - int(ROIwidth/2)-1 : cross_y + int(ROIwidth/2), cross_x + ROIoffset-int(ROIheight/2) -1 : cross_x + ROIoffset + int(ROIheight/2)]   #ROI_R
            ROIs[2] = self.displaycross[cross_y - ROIoffset - int(ROIheight/2) -1 :  cross_y - ROIoffset + int(ROIheight/2), cross_x-int(ROIwidth/2) -1 : cross_x + int(ROIwidth/2)]  #ROI_U
            ROIs[3] = self.displaycross[cross_y + ROIoffset - int(ROIheight/2) -1 :  cross_y + ROIoffset + int(ROIheight/2), cross_x-int(ROIwidth/2) -1 : cross_x + int(ROIwidth/2)]  #ROI_D
            
            ROIs[0] = cv2.rotate(ROIs[0],cv2.ROTATE_90_CLOCKWISE)
            ROIs[1] = cv2.rotate(ROIs[1],cv2.ROTATE_90_CLOCKWISE)
            
            for i,ROI in enumerate(ROIs):
                MTFs[i],df[i],MTF_fs[i] = MTF_crosshair.MTF_calculation(ROI, 1/pixperdeg, freqs, 50, ROIwidth)
                columnname = "Cross#{:d}_{}".format(idx+1,ROI_labels[i])
                MTFcurves[columnname] = MTFs[i]
                if i < 2:
                    direction = 'vertical'
                else:
                    direction = 'horizontal' 
                
                insert_col = [cross_x,cross_y,ROIwidth,ROIheight,direction]
                for MTF_n in MTF_fs[i]:  
                    insert_col.append(MTF_n)
                MTFinfo[columnname] =  insert_col
            
            
            #draw left ROI
            cv2.rectangle(self.displaycross, (cross_x-ROIoffset-int(ROIheight/2), cross_y - int(ROIwidth/2) ),(cross_x-ROIoffset + int(ROIheight/2), cross_y + int(ROIwidth/2) ), (30000, 30000, 30000), 2)
            #draw right ROI
            cv2.rectangle(self.displaycross, (cross_x + ROIoffset-int(ROIheight/2), cross_y - int(ROIwidth/2) ),(cross_x + ROIoffset + int(ROIheight/2), cross_y + int(ROIwidth/2) ), (30000, 30000, 30000), 2)
            #draw up ROI
            cv2.rectangle(self.displaycross, (cross_x-int(ROIwidth/2), cross_y -ROIoffset - int(ROIheight/2) ),(cross_x + int(ROIwidth/2), cross_y - ROIoffset + int(ROIheight/2) ), (30000, 30000, 30000), 2)
            #draw down ROI
            cv2.rectangle(self.displaycross, (cross_x-int(ROIwidth/2), cross_y + ROIoffset - int(ROIheight/2) ),(cross_x + int(ROIwidth/2), cross_y + ROIoffset + int(ROIheight/2) ), (30000, 30000, 30000), 2)
            
        firstcol = np.arange(0,df[0]*len(MTFs[0]),df[0])
        MTFcurves.insert(0,"Spatical frequency(lp/deg)",firstcol)
        MTFcurves.to_csv(MTF_spreadsheet)
        MTFinfo.to_csv(MTF_info)
        cv2.namedWindow('Crosshair MTF',cv2.WINDOW_NORMAL)
        cv2.imshow('Crosshair MTF',self.displaycross)
        cv2.imwrite('Crosshair with ROIs.png', self.displaycross)
        #cv2.imshow('Crosshair MTF',ROI_U)
        cv2.waitKey(0)     
        cv2.destroyAllWindows()
        
        return MTFs, df, MTF_fs, freqs
        
            
def find_files_with_extension(directory, extension):
    matching_files = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, f'*.{extension}'):
            matching_files.append(os.path.join(root, filename))
    return matching_files
    
def GRR(path, darkpath):
    if os.path.isdir(path):
        os.chdir(path)
        save_to_csv([["Path","95 prtl","5prtl","Uniformity"]], "GRR_uniformity.csv", mode = 'w')
    Listimages = find_files_with_extension(path,'bmp')
    for file in Listimages:
        obj = portable_analyzer(path)
        
        obj.load_dark(darkpath)
        obj.load_clear(file)
        #obj.clear = obj.clear - obj.darkimage   #darkfield calibration
        uniformity = obj.angular_uniformity(16,8,67.566)
        if uniformity != None:
            print("uniformity:{:.4f}".format(uniformity))
        else:
            print("uniformity:None")
        save_to_csv([[file,obj.max,obj.min,uniformity]], "GRR_uniformity.csv", mode = 'a')
        
        
def select_point(event,x,y,flags,param):
    global ix,iy
    global img
    global windowname
    if event == cv2.EVENT_LBUTTONDOWN: # captures left button double-click
        ix,iy = x,y
        print("inside select_point: ix = {:d},iy = {:d}".format(ix,iy))
        cv2.circle(img, (x,y), 8, (255, 255, 255), -1)  # Draw a red circle at the selected point
        cv2.imshow(windowname, img)

def select_point2(event,x,y,flags,param):    #pick a series of points and store them to crosscenters
    global ix,iy
    global img
    global windowname
    global crosscenters
    if event == cv2.EVENT_LBUTTONDOWN: # captures left button double-click
        crosscenters.append([x,y])
        print("inside select_point: ix = {:d},iy = {:d}".format(x,y))
        cv2.circle(img, (x,y), 8, (255, 255, 255), -1)  # Draw a red circle at the selected point
        cv2.imshow(windowname, img)
            
def pickpointfromimage(image):
    global img
    global ix, iy
    global windowname
    windowname = 'Pick the center, then click any key to quit'
    #img = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    img = image.copy()
    cv2.namedWindow(windowname,cv2.WINDOW_NORMAL)
# bind select_point function to a window that will capture the mouse click
    cv2.setMouseCallback(windowname, select_point)
    cv2.imshow(windowname,img)
    #k = cv2.waitKey(0) & 0xFF
    cv2.waitKey(0)     
    cv2.destroyAllWindows()
    print("ix = {:d},iy = {:d}".format(ix,iy))

def pickcrosshaircenters(image):
    global img
    global ix, iy
    global windowname
    global crosscenters
    windowname = 'Pick the center, then press any key to quit'
    #img = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    img = image.copy()
    cv2.namedWindow(windowname,cv2.WINDOW_NORMAL)
# bind select_point function to a window that will capture the mouse click
    cv2.setMouseCallback(windowname, select_point2)
    cv2.imshow(windowname,img)
    #k = cv2.waitKey(0) & 0xFF
    key = cv2.waitKey(0)    
    cv2.destroyAllWindows()
    for point in crosscenters:
        print("Cross center X: {:d} Cross center Y: {:d}".format(point[0],point[1]))
    #print("ix = {:d},iy = {:d}".format(ix,iy))        

def save_to_csv(data, file_name, mode = 'w'):
    with open(file_name, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

def load_json(filename):
    with open(filename, "r") as f:
        configs = json.load(f)
    return configs
    
def find_string_between(source_string, start_string, end_string):
    pattern = re.escape(start_string) + "(.*?)" + re.escape(end_string)
    match = re.search(pattern, source_string)
    
    if match:
        return match.group(1)
    else:
        return None
    

#Tim's functions for checkerboard detection
def extrapolate_polyfit(src_index, dst_index,src_y, degree=3):
    """
    Performs a polynomial fit of the given degree and extrapolates the values for a new array of x values.

    Args:
        src_index array (numpy.ndarray): e.g. np.arange(1, 16).
        dst_index array (numpy.ndarray): e.g. np.arange(0, 17).
        src_y (numpy.ndarray): The y values.
        degree (int): The degree of the polynomial fit.

    Returns:
        numpy.ndarray: The extrapolated y values for the new array of x values.
    """
    assert len(src_index) == len(src_y), "x and y must have the same length"

    # perform a polynomial fit of the given degree
    coefficients = np.polyfit(src_index, src_y, degree)

    # calculate the corresponding y values using the polynomial fit
    dst_y = np.polyval(coefficients, dst_index)

    return dst_y

def image_enhancement_with_bg(img_src, img_bg):
    img_src8=(img_src/256).astype('uint8')
    img_bg8=(img_bg/256).astype('uint8')

    imgdiv = cv2.divide(img_src8, img_bg8, scale=255)
    cv2.imwrite('enhancedimage.png',imgdiv)
    return imgdiv

def get_checkerboard_center(src_img, cb_full_size, cb_detectable_size ,detectable_offset, degree=3):
    #ret, corners, meta = cv2.findChessboardCornersSBWithMeta(src_img, cb_detectable_size, cv2.CALIB_CB_LARGER + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret, corners, meta = cv2.findChessboardCornersSBWithMeta(src_img, cb_detectable_size, None)
    #ret, corners = cv2.findChessboardCorners(src_img, cb_detectable_size, None)
    
    ##Tim's addition for checker orientation
    corners.reshape(cb_detectable_size[1],cb_detectable_size[0],2)
    if (corners[1,0,1] > corners[-1,-1,1]):
        print("flip detected corners vertically")
        corners=np.flip(corners, 0)
    if (corners[1,0,0] > corners[-1,-1,0]):
        print("flip detected corners horizontally")
        corners=np.flip(corners, 1)
    
    corners=corners.reshape(cb_detectable_size[1]*cb_detectable_size[0],1,2)
    
    ##End of Tim's addition for checker orientation
    
    cv2.imwrite(f'{detectable_offset}output_image_normalized.png', src_img)    
    imgcolor = cv2.cvtColor(src_img,cv2.COLOR_GRAY2RGB)
    
    #pdb.set_trace()  #breakpoint
    
    cv2.drawChessboardCorners(imgcolor, meta.shape[::-1], corners, ret)
    #cv2.drawChessboardCorners(imgcolor,cb_detectable_size, corners, ret)
    
    cv2.imwrite(f'{detectable_offset}output_image_chessboard.png', imgcolor)

    #get grid left and right interpolation
    dst_corners=[]
    merged_xy_left_right = np.array([])
    merged_xy_all = np.array([])
    src_corners_x=[]
    src_corners_y=[]
    grid_x=cb_detectable_size[0]
    grid_y=cb_detectable_size[1]
    for subarray in corners:
        src_corners_x.append(subarray[0][0])
        src_corners_y.append(subarray[0][1])
    for i in range(0,grid_y):  # extrapolate x y for left and right boundary
        corners_x=src_corners_x[i*grid_x+0:i*grid_x+grid_x]
        corners_y=src_corners_y[i*grid_x+0:i*grid_x+grid_x]
        dst_x = extrapolate_polyfit(np.arange(1,grid_x+1), np.arange(0,grid_x+2), corners_x, degree=3)
        dst_y = extrapolate_polyfit(np.arange(1,grid_x+1), np.arange(0,grid_x+2), corners_y, degree=3)
        merged_xy = np.array([[dst_x[i], dst_y[i]] for i in range(len(dst_x))])
        merged_xy_left_right= np.append(merged_xy_left_right,merged_xy)

    merged_xy_left_right = merged_xy_left_right.reshape(((grid_y)*(grid_x+2), 1, 2))
    merged_xy_left_right = merged_xy_left_right.astype(np.float32)
    src_corners_x=[]
    src_corners_y=[]
    for subarray in merged_xy_left_right:
        src_corners_x.append(subarray[0][0])
        src_corners_y.append(subarray[0][1])
    for j in range(0,cb_full_size[1]+2):  # extrapolate x y for top and bottom boundary
        corners_x=[src_corners_x[i] for i in range(j, len(src_corners_x), (cb_full_size[0]+2))]
        corners_y=[src_corners_y[i] for i in range(j, len(src_corners_y), (cb_full_size[1]+2))]
        dst_x = extrapolate_polyfit(np.arange(detectable_offset+1,detectable_offset+1+grid_y), np.arange(0,cb_full_size[0]+2), corners_x, degree=3)
        dst_y = extrapolate_polyfit(np.arange(detectable_offset+1,detectable_offset+1+grid_y), np.arange(0,cb_full_size[1]+2), corners_y, degree=3)
        merged_xy = np.array([[dst_x[i], dst_y[i]] for i in range(len(dst_x))])
        merged_xy_all= np.append(merged_xy_all,merged_xy)

    merged_xy_all = merged_xy_all.reshape(((cb_full_size[0]+2)*(cb_full_size[1]+2), 1, 2))
    merged_xy_all= merged_xy_all.astype(np.float32)
    imgcolor = cv2.cvtColor(src_img,cv2.COLOR_GRAY2RGB)
    cv2.drawChessboardCorners(imgcolor, cb_full_size, merged_xy_all, ret)
    cv2.imwrite(f'{detectable_offset}output_image_chessboard_extrapolate.png', imgcolor)
    imgcolor = cv2.cvtColor(src_img,cv2.COLOR_GRAY2RGB)
    ## Radius of circle 
    radius = 5
   
    ## Blue color in BGR 
    color = (0, 255, 0) 
   
    # Line thickness of 2 px 
    thickness = 2
   
    #merged_center = np.array([])
    output_list = []
    # Using cv2.circle() method 
    # Draw a circle with blue line borders of thickness of 2 px 
    nrow = cb_full_size[1]
    ncol = cb_full_size[0]
    for j in range(0,nrow+1):
        for i in range(0,ncol+1):
           # coor=0.25*(merged_xy_all[j*(ncol+2)+i,0,:]+merged_xy_all[j*(ncol+2)+i+1,0,:]+merged_xy_all[j*(ncol+2)+i+(ncol+2),0,:]+merged_xy_all[j*(ncol+2)+i+(ncol+2)+1,0,:])
            coor=0.25*(merged_xy_all[j*(ncol+2)+i,0,:]+merged_xy_all[j*(ncol+2)+i+1,0,:]+merged_xy_all[j*(ncol+2)+i+(ncol+2),0,:]+merged_xy_all[j*(ncol+2)+i+(ncol+2)+1,0,:])
            #merged_center = np.append(merged_center, coor)
            output_list.append(tuple(coor))
#    merged_center = merged_center.reshape(((cb_full_size[0]+1)*(cb_full_size[1]+1), 1, 2))
#    merged_center= merged_center.astype(np.float32)
    
            imgcolor = cv2.circle(imgcolor, np.intp(coor), radius, color, thickness) 
            index =j*ncol+i
            imgcolor = cv2.putText(imgcolor, f'{index}', np.intp(coor),cv2.FONT_HERSHEY_SIMPLEX,  0.3, (0,0, 255), 1)
    cv2.imwrite(f'{detectable_offset}output_image_center.png', imgcolor)
    return output_list # merged_center
    
def swap(coordinate):
    temp = coordinate[0]
    coordinate[0] = coordinate[1]
    coordinate[1] = temp
    return coordinate
    
def get_data_within_circle(array, center, radius):
    """
    Get data within a circle from a 2D array.

    Parameters:
    - array: 2D NumPy array containing the data.
    - center: Tuple (x, y) representing the center of the circle.
    - radius: Radius of the circle.

    Returns:
    - data_within_circle: NumPy array containing data within the circle.
    """
    # Create a grid of coordinates
    rows, cols = np.indices(array.shape)
    #coordinates = np.column_stack((rows.flatten(), cols.flatten()))
    coordinates = np.column_stack((cols.flatten(), rows.flatten()))
    # Calculate distances from each point to the center
    distances = np.linalg.norm(coordinates - center, axis=1)

    # Mask points within the circle based on the radius
    mask_within_circle = distances <= radius

    # Reshape the mask to match the shape of the original array
    mask_within_circle = mask_within_circle.reshape(array.shape)

    # Extract data within the circle
    data_within_circle = array[mask_within_circle]

    return data_within_circle        

def transpose_list_to_csv(input_list, output_file):
    # Transpose the list
    transposed_data = list(map(list, zip(*input_list)))
    
    # Write the transposed data to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the transposed data rows
        csv_writer.writerows(transposed_data)
    
    #print("Transposed data has been saved to", output_file)

if __name__ == "__main__":
    folderpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708'
    darkpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\DarkField\20230708200717010_0.bmp'
    checkerpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\20230708192640701_0.bmp'
    negcheckerpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\20230708192650843_0.bmp'
    slbcheckerpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\Baseline\20230708220725306_0.bmp'
    slbnegcheckerpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\Baseline\20230708220720963_0.bmp'
    slbclearpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\Baseline\20230708220716516_0.bmp'
    Ghostpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\20230708192702672_0.bmp'
    clearpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\20230708192627321_0.bmp'
    crosshairpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\Test1_Sample1A_DefaultCameraSettings\Baseline\20230708220811629_0.bmp'
    crosscoordinate = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\20230708\crosshair centers coordinate.json'
    gamma_path = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\scripts\See3Cam160_GammCurve.xlsx'
    #ND_A_path
    #ND_B_path
    manualload_cross_coordinate = True
    
    GRRpath = r'G:\Shared drives\Display & Optics Drive - Archive\Waveguide\Waveguide Metrology\Portable WGtester\data\07202023\GRR'
    
    obj = portable_analyzer(folderpath)
    obj.load_gamma(gamma_path)   #load gamma curve for gamma correction, disable this statement if no need for gamma correction
    obj.load_dark(darkpath)
    obj.load_clear(clearpath)
    obj.load_checker_negchecker(checkerpath,negcheckerpath)
    obj.load_ghost(Ghostpath)
    obj.load_SLB(slbcheckerpath, slbnegcheckerpath, slbclearpath)
    obj.load_cross(crosshairpath)
    
    
    
    
    
    # if manualload_cross_coordinate == True:
    #     pickcrosshaircenters(obj.cross)
    # else:
    #     crosscenters = load_json(crosscoordinate)
    
    # MTFs, df, MTF_fs, freqs = obj.caculate_crosshair_MTF()
    
    #efficiency calculation with clear pattern:
    obj.efficiency(FOVx = 16, FOVy = 8, pixperdeg = 67.566,exposure_SLB = 1, exposure_DUT = 1,center_m = 2222, center_n = 1649, ND_A_path = None, ND_B_path = None)
    print("Efficieny is {:.4f}".format(obj.eff))
    save_to_csv(obj.eff_map, "Efficiency_map.csv", mode = 'w')
    
    cv2.namedWindow("SLB_clear",cv2.WINDOW_NORMAL)
    cv2.imshow("SLB_clear",obj.slbclear_pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('SLB_clear_efficiencyFOV.png',obj.slbclear_pattern)
    
    cv2.namedWindow("DUT_clear",cv2.WINDOW_NORMAL)
    cv2.imshow("DUT_clear",obj.clear_pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('clear_efficiencyFOV.png',obj.clear_pattern)
    
    
    
    
    
    
    pickpointfromimage(obj.slbchecker)
    #obj.SLB_contrast(10, 8, squaresize_m = 127, squaresize_n = 127,  center_m = 2315, center_n = 1495)
    
    obj.SLB_contrast(10, 8, squaresize_m = 127, squaresize_n = 127,  center_m = ix, center_n = iy)
    
    #obj.meancontrast(10, 8, squaresize_m = 127, squaresize_n = 127,  center_m = 2199, center_n = 1800)
    
    pickpointfromimage(obj.checker)
    
    obj.meancontrast(10, 8, squaresize_m = 127, squaresize_n = 127,  center_m = ix, center_n = iy)
    obj.contrastcalibration()
    
    print("SLB Checkerboard mean contrast is {:.5f}".format(obj.slb_meancontrast))
    print("SLB Checkerboard harmonic mean contrast is {:.5f}".format(obj.slb_hmeancontrast))
    print("Ucalibrated Checkerboard mean contrast is {:.5f}".format(obj.meancontrast))
    print("Ucalibrated Checkerboard harmonic mean contrast is {:.5f}".format(obj.hmeancontrast))
    print("Calibrated Checkerboard mean contrast is {:.5f}".format(obj.calibrated_meancontrast))
    print("Calibrated Checkerboard harmonic mean contrast is {:.5f}".format(obj.calibrated_hmeancontrast))
    
    
    cv2.namedWindow("SLB_checkerimage",cv2.WINDOW_NORMAL)
    cv2.imshow("SLB_checkerimage",obj.slbchecker)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('SLB_checker.png',obj.checker)
    
    cv2.namedWindow("checkerimage",cv2.WINDOW_NORMAL)
    cv2.imshow("checkerimage",obj.checker)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('checker.png',obj.checker)
    
    #Measure ghost
    obj.Ghost()
    cv2.imwrite('Ghost.png',obj.displayghost)
    obj.ghostdata.to_csv('Ghostdata.csv')
    save_to_csv(obj.slb_contrastcsv, "SLB_contrastmap.csv", mode = 'w')
    save_to_csv(obj.contrastcsv, "Ucalibrated_contrastmap.csv", mode = 'w')
    save_to_csv(obj.calibrated_contrastcsv, "Calibrated_contrastmap.csv", mode = 'w')
    
    meandata = [
        ["Mean contrast",obj.meancontrast],
        ["Harmonic mean contrast",obj.hmeancontrast]
        ]
    save_to_csv(meandata, "Ucalibrated_contrastmap.csv", mode = 'a')
    
    slbmeandata = [
        ["SLB_Meancontrast",obj.slb_meancontrast],
        ["SLB_Harmonic mean contrast",obj.slb_hmeancontrast]
        ]
    save_to_csv(slbmeandata, "SLB_contrastmap.csv", mode = 'a')
    
    calibrated_meandata = [
        ["SLB_Meancontrast",obj.calibrated_meancontrast],
        ["SLB_Harmonic mean contrast",obj.calibrated_hmeancontrast]
        ]
    save_to_csv(calibrated_meandata, "Calibrated_contrastmap.csv", mode = 'a')
    
    # obj.angular_uniformity(16,8,67.566)
    # GRR(GRRpath,darkpath)
