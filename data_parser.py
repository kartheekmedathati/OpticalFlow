import os
from os.path import *
import numpy as np
from glob import glob
from pathlib import Path

## Main file for the interactive visualization
from load_config import load_config
cfg=load_config()

def GetMpiSintel(cfg):
    flow_root = join(cfg['data_dir'],'MPISintel/training/flow')
    image_root = join(cfg['data_dir'],'MPISintel/training/final')
    file_list = sorted(glob(join(flow_root, '*/*.flo')))

    flow_list = []
    image_list = []

    for file in file_list:
        if 'test' in file:
            # print file
            continue

        fbase = file[len(flow_root)+1:]
        fprefix = fbase[:-8]
        fnum = int(fbase[-8:-4])

        img1 = join(image_root, fprefix + "%04d" % (fnum+0) + '.png')
        img2 = join(image_root, fprefix + "%04d" % (fnum+1) + '.png')

        if not isfile(img1) or not isfile(img2) or not isfile(file):
            continue

        image_list += [[img1, img2]]
        flow_list += [file]

    assert (len(image_list) == len(flow_list))
    return image_list,flow_list


def GetWallach(cfg):
    flow_root =  join(cfg['data_dir'],'Stimuli/Hyderabad') 
    #print("Root directory: ", flow_root)
    file_list = sorted(glob(join(flow_root, '*/*.flo')))
    
    flow_list = []
    image_list = []

    for file in file_list:
        if 'test' in file:
            # print file
            continue

        fbase = file[len(flow_root)+1:]
        fprefix = fbase[:-6]
        fnum = int(fbase[-6:-4])
        head, tail = os.path.split(fprefix)
        fprefix = head[:-2] + "images/"
        fprefix  = fprefix + tail
        # print("-*-")
        #print("File base: ",flow_root)
        #print("fprefix: ",fprefix)
        # print("___")

        img1 = join(flow_root, fprefix + "%02d" % (fnum+0) + '.png')
        img2 = join(flow_root, fprefix + "%02d" % (fnum+1) + '.png')

        print("Img1: ",img1)
        print("Img2: ",img2)
        print("Flow: ", file)
        # print("---")
        if not isfile(img1) or not isfile(img2) or not isfile(file):
           print("Not found\n", img1)
           print("Not found\n", img2)
           exit()
           continue
        image_list += [[img1, img2]]
        flow_list += [file]

    assert (len(image_list) == len(flow_list))
    return image_list, flow_list



def GetPredictionsHornShunckWallach(cfg,compute_mode=False):
    im_list, flow_list = GetWallach(cfg)
    pred_flow_list = []
    print(im_list[0][0])
    print(im_list[0][1])
    print(flow_list[0])

    pred_root = join(cfg['data_dir'],'Stimuli/Hyderabad/pred_flow_hs')
    if not os.path.exists(pred_root):
        print("Creating results directory: ",pred_root)
        os.makedirs(pred_root)

    #_, thing_I_want = os.path.split(os.getcwd())
    method_command = join(cfg['code_dir'], 'Code3rdParty/phs_3/horn_schunck_pyramidal') 

    for i in range(len(flow_list)): # len(flow_list)
        pred_dir = join(pred_root,os.path.basename(os.path.dirname(flow_list[i])))
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        res_file = join(pred_dir,Path(flow_list[i]).name)
        pred_flow_list+= [res_file]
        if compute_mode==True:
            cmd = method_command +' '+ im_list[i][0]+' '+im_list[i][1] +' '+res_file+' '+ '0 100 4 0.5 50 0.0001 150 0'
            #print("Executing: ",cmd)
            os.system(cmd)
    return im_list,flow_list,pred_flow_list


def GetPredictionsHornShunck(cfg,compute_mode=False):
    pred_flow_list = []
    im_list, flow_list = GetMpiSintel(cfg)
    pred_root = join(cfg['data_dir'],'MPISintel/training/pred_flow_hs')
    if not os.path.exists(pred_root):
        print("Creating results directory: ",pred_root)
        os.makedirs(pred_root)

    #_, thing_I_want = os.path.split(os.getcwd())

    method_command = join(cfg['code_dir'], 'Code3rdParty/phs_3/horn_schunck_pyramidal') 

    for i in range(len(flow_list)): # len(flow_list)
        pred_dir = join(pred_root,os.path.basename(os.path.dirname(flow_list[i])))
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        res_file = join(pred_dir,Path(flow_list[i]).name)
        pred_flow_list += [res_file] 
        if compute_mode==True:
            cmd = 'time ' + method_command +' '+ im_list[i][0]+' '+im_list[i][1] +' '+res_file +' '+ '0 100 4 0.5 50 0.0001 150 0'
            print("Executing: ",cmd)
            os.system(cmd)
    return im_list, flow_list, pred_flow_list


#im_list, flow_list, pred_flow_list = GetPredictionsHornShunck(cfg, True)
# for i in range(4):
#     print(im_list[i])
#     print(flow_list[i])
#     print(pred_flow_list[i])

im_list, flow_list, pred_flow_list = GetPredictionsHornShunckWallach(cfg, True)

