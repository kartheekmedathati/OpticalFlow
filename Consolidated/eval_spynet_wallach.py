import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
from scipy.misc import imread, imresize

def getFileList(root):
	flow_root = root
        image_root = root[:-3]+'images/'
        print("Root directory: ", flow_root)
        file_list = sorted(glob(join(flow_root, '*.flo')))
        print(file_list)
        flow_list = []
        image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root):]
            fprefix = fbase[:-6]
            fnum = int(fbase[-6:-4])
            print("File base: ",fbase)
            print("Image root: ",image_root)   
             
            img1 = join(image_root, fprefix + "%02d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%02d"%(fnum+1) + '.png')
            
            print("Img1: ",img1)
            print("Img2: ",img2) 
            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            image_list += [[img1, img2]]
            flow_list += [file]
	return image_list, flow_list

if __name__ == '__main__':
	'''File list points to GT directory'''
	img_list, flow_list = getFileList('/home/MedathatiExt/OpticalFlow/Stimuli/Stimuli_Samples/moving_circle_full_r42_c255_GT/')
        for i in range(0,len(img_list)):
		print(img_list[i][0])
		print(img_list[i][1])
		print(flow_list[i])
		print("--")

