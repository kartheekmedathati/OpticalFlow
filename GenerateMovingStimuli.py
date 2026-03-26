''' Generate moving dot stimuli of different intensities and speeds, to generate at 2 frames forward and 2 frames backward from the center at every speed
Size of the image : 1280 X 760
Start with centeral pixel_Generate 2 frames backwards and 2 frames forward
'''

import os,sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2
#import cv2.cv as cv
from skimage import draw
from multiprocessing import Pool
from utils.viz_flow import viz_flow
from utils.flow_io import *
from joblib import Parallel, delayed
import pdb
import itertools
# Create un outer and inner circle. Then subtract the inner from the outer.
'''
radius = 0.15
inner_radius = radius - (stroke // 2) + (stroke % 2) - 1 
outer_radius = radius + ((stroke + 1) // 2)
XX, ci = draw.circle( 760/2,1280/2, radius=inner_radius, shape=arr.shape)
ro, co = draw.circle(760/2,1280/2, radius=outer_radius, shape=arr.shape)
arr[ro, co] = 1
arr[ri, ci] = 0
plt.imshow(arr>0)
plt.show()'''

from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc(fmt="Elapsed: %s s"):
    print(fmt % (time() - _tstart_stack.pop()))
    

def VisualizeVelocitySamples(Height, Width,vel_tsample):
    xc = Width/2 -1
    yc = Height/2-1
    contrastvalue =  255;
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    current_frame = np.zeros([Height, Width],dtype=np.float)
    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
    for VY in vel_sample:
        for VX in vel_sample:
            yc,xc = draw.circle_perimeter( int(Height/2)-1,int(Width/2)-1, radius=3, shape=current_frame.shape)
            txc = xc + VX
            tyc=  yc + VY
            current_frame[tyc,txc]= contrastvalue
            flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
            flow_gt_current[1,tyc,txc] = VY #
    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
    #plt.imshow(I_flow)
    #plt.show()
    ##
    #Writing files to disk
    cv2.imwrite('VelocitySampling.png',current_frame)
    cv2.imwrite('VelocitySampling_colorcode.png', cv2.cvtColor(I_flow, cv2.COLOR_RGB2BGR))
    
                
def GenerateMovingCircle(Height, Width,circle_radius,circ_type, vel_sample,Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    contrastvalue =  np.random.randint(1,255,size=1)[0];
    stimulustype = 'circles_'+circ_type+'_r'+str(circle_radius)+'_c'+ str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/') 
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    if not os.path.exists(os.path.dirname(Results_GT_dir)):
        os.makedirs(os.path.dirname(Results_GT_dir))
    
    #vel_tsample = np.logspace(0, np.log10(100), 10, endpoint=True)
    #vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    for VY in vel_sample: #range(-50,51):#range(-(Height/4)+1,Height/4):
        for VX in vel_sample: #range(-50,51):#range(-(Width/4)+1,Width/4):#tic()
            for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                img_name = os.path.join(Results_dir,stimulustype+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                gt_flo_name = os.path.join(Results_GT_dir,stimulustype+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                gt_flo_name_vis = os.path.join(Results_GT_dir,stimulustype+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                #Creating the buffer for the frame and flow ground truth
                current_frame = np.zeros([Height, Width],dtype=np.float)
                flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                #Computing the displacement for the pixels
                if circ_type =='perimeter':
                    yc,xc = draw.circle_perimeter( int(Height/2)-1,int(Width/2)-1, radius=circle_radius, shape=current_frame.shape) #-1 since coordinate system begins at zero
                if circ_type =='full':
                    yc,xc = draw.circle( int(Height/2)-1,int(Width/2)-1, radius=circle_radius, shape=current_frame.shape) #-1 since coordinate system begins at zero
                disp_X = VX*dispfactor
                disp_Y = VY*dispfactor
                txc = xc + disp_X
                tyc=  yc + disp_Y
                current_frame[tyc,txc] = contrastvalue;
                #Setting ground truth velocity
                flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                flow_gt_current[1,tyc,txc] = VY #
                I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                #plt.imshow(I_flow)
                #plt.show()
                ##
                #Writing files to disk
                #cv2.imwrite(img_name,current_frame)
                cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                if index<4:
	                cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
        	        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
            #toc()
            #print VX, VY




def GenerateMovingLine(Height, Width,length,orientations, vel_sample,Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    contrastvalue =  np.random.randint(1,255,1)[0];
    stimulustype = 'lines_l_'+str(length)+'_c'+ str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/')  #'/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    if not os.path.exists(os.path.dirname(Results_GT_dir)):
        os.makedirs(os.path.dirname(Results_GT_dir))
    
    #vel_tsample = np.logspace(0, np.log10(100), 10, endpoint=True)
    #vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    for current_orientation in orientations:
        for VY in vel_sample: #range(-50,51):#range(-(Height/4)+1,Height/4):
            for VX in vel_sample: #range(-50,51):#range(-(Width/4)+1,Width/4):
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    img_name = os.path.join(Results_dir,stimulustype + '_orient_'+ str(current_orientation)+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_GT_dir,stimulustype + '_orient_'+ str(current_orientation)  +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = os.path.join(Results_GT_dir,stimulustype + '_orient_'+ str(current_orientation) +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width],dtype=np.float)
                    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                    y_center_offset = np.round((length/2)*np.sin(np.radians(-1*current_orientation))).astype(int)
                    x_center_offset = np.round((length/2)*np.cos(np.radians(-1*current_orientation))).astype(int)
                    ltyc = (Height/2)-1-y_center_offset    # Generating the ends of the line using center point
                    ltxc = (Width/2)-1-x_center_offset     # SKDraw accepts r0,c0,r1,c1 for generating coordinates
                    lbyc = (Height/2) -1 + y_center_offset # We start at Center of the frame and using orientation generate end points
                    lbxc = (Width/2)-1 +x_center_offset
                    #Computing the displacement for the pixels
                    yc,xc = draw.line( int(ltyc),int(ltxc), int(lbyc),int(lbxc)) #-1 since coordinate system begins at zero
                    disp_X = VX*dispfactor
                    disp_Y = VY*dispfactor
                    txc = xc + disp_X
                    tyc=  yc + disp_Y
                    current_frame[tyc,txc] = contrastvalue;
                    #Setting ground truth velocity
                    flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc,txc] = VY #
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #plt.imshow(I_flow)
                    #plt.show()
                    ##
                    #Writing files to disk
                    #cv2.imwrite(img_name,current_frame)
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])


def GenerateMovingLine_undercircaper(Height, Width,length,circle_radius,orientations, vel_tsample,Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    contrastvalue =  np.random.randint(1,255,1)[0];
    stimulustype = 'moving_line_l__undercircaper_'+str(length)+'_c'+ str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/')  #'/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    if not os.path.exists(os.path.dirname(Results_GT_dir)):
        os.makedirs(os.path.dirname(Results_GT_dir))
    
    #vel_tsample = np.logspace(0, np.log10(100), 10, endpoint=True)
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    for current_orientation in orientations:
        for VY in vel_sample: #range(-50,51):#range(-(Height/4)+1,Height/4):
            for VX in vel_sample: #range(-50,51):#range(-(Width/4)+1,Width/4):
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    img_name = os.path.join(Results_dir,stimulustype + '_orient_'+ str(current_orientation)+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_GT_dir,stimulustype + '_orient_'+ str(current_orientation)  +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = os.path.join(Results_GT_dir,stimulustype + '_orient_'+ str(current_orientation) +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width],dtype=np.float)
                    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                    y_center_offset = np.round((length/2)*np.sin(np.radians(-1*current_orientation))).astype(int)
                    x_center_offset = np.round((length/2)*np.cos(np.radians(-1*current_orientation))).astype(int)
                    ltyc = (Height/2)-1-y_center_offset    # Generating the ends of the line using center point
                    ltxc = (Width/2)-1-x_center_offset     # SKDraw accepts r0,c0,r1,c1 for generating coordinates
                    lbyc = (Height/2) -1 + y_center_offset # We start at Center of the frame and using orientation generate end points
                    lbxc = (Width/2)-1 +x_center_offset
                    #Computing the displacement for the pixels
                    yc,xc = draw.line( ltyc,ltxc, lbyc,lbxc) #-1 since coordinate system begins at zero
                    disp_X = VX*dispfactor
                    disp_Y = VY*dispfactor
                    txc = xc + disp_X
                    tyc=  yc + disp_Y
                    current_frame[tyc,txc] = contrastvalue;
                    #adding circular aperture                    
                    yc_circ,xc_circ = draw.circle_perimeter( Height/2-1,Width/2-1, radius=circle_radius, shape=current_frame.shape)
                    current_frame[yc_circ,xc_circ] = contrastvalue;                    
                    #Setting ground truth velocity
                    flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc,txc] = VY #
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #plt.imshow(I_flow)
                    #plt.show()
                    ##
                    #Writing files to disk
                    #cv2.imwrite(img_name,current_frame)
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])

def GenerateMovingRectangles(Height, Width,length,aspectratio,orientations, vel_sample,Results_base_dir):
    #vel_tsample = np.logspace(0, np.log10(100), 10, endpoint=True)
    #vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    xc = Width/2 -1
    yc = Height/2-1
    contrastvalue =  np.random.randint(1,255,1)[0];
    stimulustype = 'rectangles_l_'+str(length)+'_aspr_'+str(aspectratio)+'_c'+ str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/')  #'/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    if not os.path.exists(os.path.dirname(Results_GT_dir)):
        os.makedirs(os.path.dirname(Results_GT_dir))
    
    for current_orientation in orientations:
        for VY in vel_sample: #range(-50,51):#range(-(Height/4)+1,Height/4):
            for VX in vel_sample: #range(-50,51):#range(-(Width/4)+1,Width/4):
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    img_name = os.path.join(Results_dir,stimulustype + '_orient_'+ str(current_orientation)+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_GT_dir,stimulustype + '_orient_'+ str(current_orientation)  +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = os.path.join(Results_GT_dir,stimulustype + '_orient_'+ str(current_orientation) +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width],dtype=np.float)
                    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                    y_center_offset1 = np.round((length/2)*np.sin(np.radians(-1*current_orientation))).astype(int)
                    x_center_offset1 = np.round((length/2)*np.cos(np.radians(-1*current_orientation))).astype(int)
                    y_center_offset2 = np.round((aspectratio*length/2)*np.sin(np.radians(-1*(current_orientation+90)))).astype(int)
                    x_center_offset2 = np.round((aspectratio*length/2)*np.cos(np.radians(-1*(current_orientation+90)))).astype(int)
                    rtyc = np.zeros(4)
                    rtxc = np.zeros(4)

                    rtyc[0] = (Height/2)-1-y_center_offset1 -y_center_offset2
                    rtyc[1] = (Height/2)-1-y_center_offset1 +y_center_offset2
                    rtyc[3] = (Height/2)-1+y_center_offset1 -y_center_offset2
                    rtyc[2] = (Height/2)-1+y_center_offset1 +y_center_offset2

                    rtxc[0] = (Width/2)-1-x_center_offset1 -x_center_offset2
                    rtxc[1] = (Width/2)-1-x_center_offset1 +x_center_offset2
                    rtxc[3] = (Width/2)-1+x_center_offset1 -x_center_offset2
                    rtxc[2] = (Width/2)-1+x_center_offset1 +x_center_offset2

                    #pdb.set_trace()
                    #Computing the displacement for the pixels
                    yc,xc = draw.polygon( rtyc,rtxc,shape=current_frame.shape) #-1 since coordinate system begins at zero
                    disp_X = VX*dispfactor
                    disp_Y = VY*dispfactor
                    txc = xc + disp_X
                    tyc=  yc + disp_Y
                    current_frame[tyc,txc] = contrastvalue;
                    #Setting ground truth velocity
                    flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc,txc] = VY #
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #plt.imshow(I_flow)
                    #plt.show()
                    ##
                    #Writing files to disk
                    #cv2.imwrite(img_name,current_frame)
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    
def GenerateGrating(Height, Width,sinSf,sinDirection,velocity):
    yc = np.linspace(0,Height,Height,endpoint=False)-np.floor(Height/2)
    xc = np.linspace(0,Width,Width, endpoint=False) -np.floor(Width/2)
    t = np.linspace(0,5,5,endpoint=False)
    [XX, YY,tt] = np.meshgrid(xc,yc,t)
    YY = -YY;
    sinTf = velocity*sinSf  # v = ft x lambda -> ft = v/lambda -> ft = v x fspatial  
    res = np.cos(2*np.pi*sinSf*np.cos(np.deg2rad(sinDirection))*XX + 2*np.pi*sinSf*np.sin(np.deg2rad(sinDirection))*YY -2*np.pi*sinTf*tt);
    res = np.multiply(res,0.5)+0.5
    return res

def GeneratePlaid(Height, Width,sinSf1,sinDirection1,velocity1, sinSf2, sinDirection2, Velocity2):
    firstGrating =  GenerateGrating(Height, Width, sinSf1, sinDirection1, velocity1);
    secondGrating = GenerateGrating(Height, Width, sinSf2, sinDirection2, velocity1);
    plaidpattern = np.multiply(firstGrating,0.5) + np.multiply(secondGrating,0.5);
    return plaidpattern

def GenerateMovingPlaids(Height,Width,Radius, frequencies, orientations, orientation_offset, Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    stimulustype = 'plaids_circ_aper_R_'+str(Radius)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/') 
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')

    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    if not os.path.exists(os.path.dirname(Results_GT_dir)):
        os.makedirs(os.path.dirname(Results_GT_dir))
    #frequencies = np.divide(1.0,range(2,2*Radius,8))
    #frequencies = np.divide(1.0,range(2,2*Radius,8))
    
    for stimfreqies in itertools.combinations(frequencies,2):
        for orientations_iter in itertools.product(orientations,orientation_offset):
            current_orientations = np.asarray(orientations_iter)
            current_orientations[1] = current_orientations[1]+current_orientations[0]
            vel_sample = np.logspace(0, np.log10(0.5/np.max((stimfreqies[0],stimfreqies[1]))), 8, endpoint=True)   
            #vel_sample = [2.0,3.0]
            #vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
            tic()            
            for current_speeds in itertools.combinations(vel_sample,2):
                contrastvalue =  np.random.randint(10,255,1)[0];
                a = GeneratePlaid(Height,Width, stimfreqies[0],current_orientations[0],current_speeds[0],stimfreqies[1],current_orientations[1], current_speeds[1])
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width],dtype=np.float)
                    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                    tyc,txc = draw.circle( Height/2-1,Width/2-1, radius=Radius, shape=current_frame.shape)
                    current_frame[tyc,txc] = contrastvalue*a[tyc,txc,index]

                    ## Compute IoC
                    
                    if current_speeds[0]==current_speeds[1]:
                        flow_gt_current[0,tyc,txc] = 0.5*((current_speeds[0]*np.cos(np.deg2rad(current_orientations[0])))+ (current_speeds[1]*np.cos(np.deg2rad(current_orientations[1]))))
                        flow_gt_current[1,tyc,txc] = -1*0.5*((current_speeds[1]*np.sin(np.deg2rad(current_orientations[1])))+ (current_speeds[1]*np.sin(np.deg2rad(current_orientations[1]))))
                    else:
                        #flow_gt_current[0,tyc,txc] = (current_speeds[0]*np.sin(np.deg2rad(current_orientations[0]))-current_speeds[1]*np.sin(np.deg2rad(current_orientations[1])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))   
                        #flow_gt_current[1,tyc,txc] = (current_speeds[0]*np.cos(np.deg2rad(current_orientations[0]))-current_speeds[1]*np.cos(np.deg2rad(current_orientations[1])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))
                        flow_gt_current[0,tyc,txc] = (current_speeds[0]*np.sin(np.deg2rad(current_orientations[1]))-current_speeds[1]*np.sin(np.deg2rad(current_orientations[0])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))   
                        flow_gt_current[1,tyc,txc] = -1*(current_speeds[0]*np.cos(np.deg2rad(current_orientations[1]))-current_speeds[1]*np.cos(np.deg2rad(current_orientations[0])))/np.sin(np.deg2rad(current_orientations[0])-np.deg2rad(current_orientations[1]))
                    VX = np.round(flow_gt_current[0,tyc[0],txc[0]])
                    VY = np.round(flow_gt_current[1,tyc[0],txc[0]])
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #Writing files to disk
                    img_name = os.path.join(Results_dir,stimulustype + '_cyc_'+str(np.round(2*Radius*stimfreqies[0]))+ '_' + str(np.round(2*Radius*stimfreqies[1])) + \
                        '_orient_' + str(current_orientations[0]) + '_' + str(current_orientations[1]) + \
                        '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__' + str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_GT_dir,stimulustype+ '_cyc_'+str(np.round(2*Radius*stimfreqies[0])) + '_' + str(np.round(2*Radius*stimfreqies[1])) + \
                        '_orient_' + str(current_orientations[0]) + '_' + str(current_orientations[1]) + \
                        '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = gt_flo_name[:-4]+'_GT.png'
                    #cv2.imwrite(img_name,current_frame)
                    #cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
            toc


def GenerateMovingHybridPlaids(Height,Width,Radius, frequencies, orientations, orientation_offset,grating_offset, Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    stimulustype = 'hybridplaids_circ_aper_R_'+str(Radius)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/') 
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')

    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    if not os.path.exists(os.path.dirname(Results_GT_dir)):
        os.makedirs(os.path.dirname(Results_GT_dir))
    #frequencies = np.divide(1.0,range(2,2*Radius,8))
    #frequencies = np.divide(1.0,range(2,2*Radius,8))
    
    for stimfreqies in itertools.combinations(frequencies,2):
        for orientations_iter in itertools.product(orientations,orientation_offset):
            current_orientations = np.asarray(orientations_iter)
            current_orientations[1] = current_orientations[1]+current_orientations[0]
            vel_sample = np.logspace(0, np.log10(0.5/np.max((stimfreqies[0],stimfreqies[1]))), 8, endpoint=True)   
            #vel_sample = [2.0,3.0]
            #vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
            tic()            
            for current_speeds in itertools.combinations(vel_sample,2):
                contrastvalue =  np.random.randint(10,255,1)[0]
                firstGrating =  GenerateGrating(Height, Width, stimfreqies[0], current_orientations[0],current_speeds[0])
                secondGrating = GenerateGrating(Height, Width, stimfreqies[1],current_orientations[1], current_speeds[1])
                a = GeneratePlaid(Height,Width, stimfreqies[0],current_orientations[0],current_speeds[0],stimfreqies[1],current_orientations[1], current_speeds[1])
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width],dtype=np.float)
                    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                    ## Left grating
                    tyc_lg,txc_lg = draw.circle( Height/2-1,Width/2-1-grating_offset, radius=Radius, shape=current_frame.shape)
                    current_frame[tyc_lg,txc_lg] = contrastvalue*firstGrating[tyc_lg,txc_lg,index]
                    flow_gt_current[0,tyc_lg,txc_lg] = current_speeds[0]*np.cos(np.deg2rad(current_orientations[0])) #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc_lg,txc_lg] = -1*current_speeds[0]*np.sin(np.deg2rad(current_orientations[0]))
                    ## Right grating
                    tyc_rg,txc_rg = draw.circle( Height/2-1,Width/2-1+grating_offset, radius=Radius, shape=current_frame.shape)
                    current_frame[tyc_rg,txc_rg] = contrastvalue*secondGrating[tyc_rg,txc_rg,index]
                    flow_gt_current[0,tyc_rg,txc_rg] = current_speeds[1]*np.cos(np.deg2rad(current_orientations[1])) #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc_rg,txc_rg] = -1*current_speeds[1]*np.sin(np.deg2rad(current_orientations[1]))

                    ## Intersection
                    lg_c = set((tuple(zip(tyc_lg,txc_lg))))
                    rg_c = set((tuple(zip(tyc_rg,txc_rg))))

                    int_c = set.intersection(lg_c,rg_c)
                    tyc,txc = zip(*int_c)
                    tyc = np.array(tyc)
                    txc = np.array(txc)
                    #tyc,txc = draw.circle( Height/2-1,Width/2-1, radius=Radius, shape=current_frame.shape)
                    
                    
                    current_frame[tyc,txc] = contrastvalue*a[tyc,txc,index]

                    ## Compute IoC
                    
                    if current_speeds[0]==current_speeds[1]:
                        flow_gt_current[0,tyc,txc] = 0.5*((current_speeds[0]*np.cos(np.deg2rad(current_orientations[0])))+ (current_speeds[1]*np.cos(np.deg2rad(current_orientations[1]))))
                        flow_gt_current[1,tyc,txc] = -1*0.5*((current_speeds[1]*np.sin(np.deg2rad(current_orientations[1])))+ (current_speeds[1]*np.sin(np.deg2rad(current_orientations[1]))))
                    else:
                        #flow_gt_current[0,tyc,txc] = (current_speeds[0]*np.sin(np.deg2rad(current_orientations[0]))-current_speeds[1]*np.sin(np.deg2rad(current_orientations[1])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))   
                        #flow_gt_current[1,tyc,txc] = (current_speeds[0]*np.cos(np.deg2rad(current_orientations[0]))-current_speeds[1]*np.cos(np.deg2rad(current_orientations[1])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))
                        flow_gt_current[0,tyc,txc] = (current_speeds[0]*np.sin(np.deg2rad(current_orientations[1]))-current_speeds[1]*np.sin(np.deg2rad(current_orientations[0])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))   
                        flow_gt_current[1,tyc,txc] = -1*(current_speeds[0]*np.cos(np.deg2rad(current_orientations[1]))-current_speeds[1]*np.cos(np.deg2rad(current_orientations[0])))/np.sin(np.deg2rad(current_orientations[0])-np.deg2rad(current_orientations[1]))
                    VX = np.round(flow_gt_current[0,tyc[0],txc[0]])
                    VY = np.round(flow_gt_current[1,tyc[0],txc[0]])
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #Writing files to disk
                    img_name = os.path.join(Results_dir,stimulustype + '_cyc_'+str(np.round(2*Radius*stimfreqies[0]))+ '_' + str(np.round(2*Radius*stimfreqies[1])) + \
                        '_orient_' + str(current_orientations[0]) + '_' + str(current_orientations[1]) + \
                        '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__' + str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_GT_dir,stimulustype+ '_cyc_'+str(np.round(2*Radius*stimfreqies[0])) + '_' + str(np.round(2*Radius*stimfreqies[1])) + \
                        '_orient_' + str(current_orientations[0]) + '_' + str(current_orientations[1]) + \
                        '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = gt_flo_name[:-4]+'_GT.png'
                    #cv2.imwrite(img_name,current_frame)
                    #cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
            toc


# def GenerateMovingPlaids(Height,Width,Radius, orientations, Results_base_dir):
#     xc = Width/2 -1
#     yc = Height/2-1
#     contrastvalue =  np.random.randint(10,255,1)[0];
#     stimulustype = 'moving_plaids_circ_aper_R_'+str(Radius)+'_c'+ str(contrastvalue)
#     Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/') 
#     Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
#     if not os.path.exists(os.path.dirname(Results_dir)):
#         os.makedirs(os.path.dirname(Results_dir))
#     if not os.path.exists(os.path.dirname(Results_GT_dir)):
#         os.makedirs(os.path.dirname(Results_GT_dir))
#     #frequencies = np.divide(1.0,range(2,2*Radius,8))
#     #frequencies = np.divide(1.0,range(2,2*Radius,8))
#     frequencies = [1.0/3,1.0/5,1.0/7,1.0/10,1.0/20,1.0/30]
#     for stimfreqies in itertools.combinations(frequencies,2):
#         for current_orientations in itertools.combinations(orientations,2):
#             vel_tsample = np.logspace(0, np.log10(0.5/np.max((stimfreqies[0],stimfreqies[1]))), 8, endpoint=True)   
#             vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
#             tic()            
#             for current_speeds in itertools.combinations(vel_sample,2):
#                 a = GeneratePlaid(Height,Width, stimfreqies[0],current_orientations[0],current_speeds[0],stimfreqies[1],current_orientations[1], current_speeds[1])
#                 for index, dispfactor in enumerate([-2, -1, 0,1,2]):
#                     #Creating the buffer for the frame and flow ground truth
#                     current_frame = np.zeros([Height, Width],dtype=np.float)
#                     flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
#                     tyc,txc = draw.circle( Height/2-1,Width/2-1, radius=Radius, shape=current_frame.shape)
#                     current_frame[tyc,txc] = contrastvalue*a[tyc,txc,index]
#                     ## Compute IoC
                     
#                     flow_gt_current[0,tyc,txc] = (current_speeds[0]*np.sin(np.deg2rad(current_orientations[0]))-current_speeds[1]*np.sin(np.deg2rad(current_orientations[1])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))
                        
#                     flow_gt_current[1,tyc,txc] = (current_speeds[0]*np.cos(np.deg2rad(current_orientations[0]))-current_speeds[1]*np.cos(np.deg2rad(current_orientations[1])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))
#                     #flow_gt_current[0,tyc,txc] = current_speed*np.cos(np.deg2rad(current_orientation)) #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
#                     #flow_gt_current[1,tyc,txc] = current_speed*np.sin(np.deg2rad(current_orientation)) #
#                     VX = np.round(flow_gt_current[0,tyc[0],txc[0]])
#                     VY = np.round(flow_gt_current[1,tyc[0],txc[0]])
#                     I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
#                     #Writing files to disk
#                     img_name = os.path.join(Results_dir,stimulustype + '_cyc_'+str(np.round(2*Radius*stimfreqies[0]))+ '_' + str(np.round(2*Radius*stimfreqies[1])) + \
#                         '_orient_' + str(current_orientations[0]) + '_' + str(current_orientations[1]) + \
#                         '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__' + str(index).zfill(2)+'.png')
#                     gt_flo_name = os.path.join(Results_GT_dir,stimulustype+ '_cyc_'+str(np.round(2*Radius*stimfreqies[0])) + '_' + str(np.round(2*Radius*stimfreqies[1])) + \
#                         '_orient_' + str(current_orientations[0]) + '_' + str(current_orientations[1]) + \
#                         '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
#                     gt_flo_name_vis = gt_flo_name[:-4]+'.png'
#                     #cv2.imwrite(img_name,current_frame)
#                     cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
#                     if index<4:
#                         cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
#                         flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
#             toc

def GenerateMovingBarberPole(Height, Width, rect_length, rect_width, frequencies, orientations, aperture_orientations, Results_base_dir):
    """Generate barber pole stimuli: a grating viewed through an elongated rectangular aperture.

    The barber pole illusion: a grating's perceived motion shifts toward the long axis
    of the aperture. Ground truth provides both the component velocity (normal to grating)
    and the barber-pole-predicted velocity (along aperture long axis).

    For each grating, we store:
      - Layer 0 (component): velocity normal to the grating orientation
      - Layer 1 (barberpole): velocity along the aperture's long axis, with magnitude
        equal to speed / cos(angle between grating normal and aperture axis)
    """
    contrastvalue = np.random.randint(10, 255, 1)[0]
    stimulustype = 'barberpole_l'+str(rect_length)+'_w'+str(rect_width)+'_c'+str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir, stimulustype+'_images/')
    Results_GT_dir = os.path.join(Results_base_dir, stimulustype+'_GT/')
    Results_GT_bp_dir = os.path.join(Results_base_dir, stimulustype+'_GT_barberpole/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    if not os.path.exists(os.path.dirname(Results_GT_dir)):
        os.makedirs(os.path.dirname(Results_GT_dir))
    if not os.path.exists(os.path.dirname(Results_GT_bp_dir)):
        os.makedirs(os.path.dirname(Results_GT_bp_dir))

    for stimfreq in frequencies:
        for grating_orient in orientations:
            vel_tsample = np.logspace(0, np.log10(0.5/stimfreq), 8, endpoint=True)
            vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1], [0], vel_tsample), axis=0))
            for current_speed in vel_sample:
                for aper_orient in aperture_orientations:
                    a = GenerateGrating(Height, Width, stimfreq, grating_orient, current_speed)
                    for index, dispfactor in enumerate([-2, -1, 0, 1, 2]):
                        current_frame = np.zeros([Height, Width], dtype=np.float64)
                        flow_gt_component = np.zeros([2, Height, Width], dtype=np.float64)
                        flow_gt_barberpole = np.zeros([2, Height, Width], dtype=np.float64)

                        # Rectangular aperture centered on image, rotated by aper_orient
                        half_l = rect_length / 2.0
                        half_w = rect_width / 2.0
                        cos_a = np.cos(np.deg2rad(-aper_orient))
                        sin_a = np.sin(np.deg2rad(-aper_orient))
                        cy, cx = Height//2 - 1, Width//2 - 1
                        corners_x = np.array([-half_l, half_l, half_l, -half_l])
                        corners_y = np.array([-half_w, -half_w, half_w, half_w])
                        rot_x = corners_x * cos_a - corners_y * sin_a + cx
                        rot_y = corners_x * sin_a + corners_y * cos_a + cy
                        mask = np.zeros([Height, Width], dtype=np.uint8)
                        pts = np.array(list(zip(rot_x.astype(int), rot_y.astype(int))), dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                        tyc, txc = np.where(mask > 0)

                        current_frame[tyc, txc] = contrastvalue * a[tyc, txc, index]

                        # Component velocity (normal to grating)
                        comp_vx = current_speed * np.cos(np.deg2rad(grating_orient))
                        comp_vy = -1 * current_speed * np.sin(np.deg2rad(grating_orient))
                        flow_gt_component[0, tyc, txc] = comp_vx
                        flow_gt_component[1, tyc, txc] = comp_vy

                        # Barber pole velocity: project onto aperture long axis
                        # Aperture long axis direction
                        ax_x = np.cos(np.deg2rad(aper_orient))
                        ax_y = -np.sin(np.deg2rad(aper_orient))
                        # Speed along aperture axis = component_speed / cos(angle_between)
                        angle_diff = np.deg2rad(grating_orient - aper_orient)
                        cos_diff = np.cos(angle_diff)
                        if abs(cos_diff) > 1e-6:
                            bp_speed = current_speed / cos_diff
                        else:
                            bp_speed = 0  # grating parallel to aperture axis
                        flow_gt_barberpole[0, tyc, txc] = bp_speed * ax_x
                        flow_gt_barberpole[1, tyc, txc] = bp_speed * ax_y

                        VX_c = np.round(comp_vx)
                        VY_c = np.round(comp_vy)
                        I_flow_comp = viz_flow(flow_gt_component[0,:,:], flow_gt_component[1,:,:])
                        I_flow_bp = viz_flow(flow_gt_barberpole[0,:,:], flow_gt_barberpole[1,:,:])

                        name_base = stimulustype + '_freq_'+str(np.round(stimfreq,4)) + \
                            '_gorient_'+str(grating_orient)+'_aorient_'+str(aper_orient) + \
                            '_VX_'+str(VX_c).zfill(4)+'_VY_'+str(VY_c).zfill(4)+'__'+str(index).zfill(2)
                        img_name = os.path.join(Results_dir, name_base+'.png')
                        gt_comp_flo = os.path.join(Results_GT_dir, name_base+'.flo')
                        gt_comp_vis = os.path.join(Results_GT_dir, name_base+'.png')
                        gt_bp_flo = os.path.join(Results_GT_bp_dir, name_base+'.flo')
                        gt_bp_vis = os.path.join(Results_GT_bp_dir, name_base+'.png')

                        cv2.imwrite(img_name, cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                        if index < 4:
                            cv2.imwrite(gt_comp_vis, cv2.cvtColor(I_flow_comp, cv2.COLOR_RGB2BGR))
                            flow_write(gt_comp_flo, flow_gt_component[0,:,:], flow_gt_component[1,:,:])
                            cv2.imwrite(gt_bp_vis, cv2.cvtColor(I_flow_bp, cv2.COLOR_RGB2BGR))
                            flow_write(gt_bp_flo, flow_gt_barberpole[0,:,:], flow_gt_barberpole[1,:,:])


def GenerateMovingBarberPlaid(Height, Width, rect_length, rect_width, frequencies, orientations, orientation_offset, aperture_orientations, Results_base_dir):
    """Generate barber plaid stimuli: a plaid pattern viewed through an elongated rectangular aperture.

    Combines the IoC computation of plaids with the aperture geometry of barber poles.
    Ground truth provides:
      - Layer 0 (IoC): the intersection-of-constraints velocity
      - Layer 1 (barberpole): velocity biased along the aperture's long axis
    """
    contrastvalue = np.random.randint(10, 255, 1)[0]
    stimulustype = 'barberplaid_l'+str(rect_length)+'_w'+str(rect_width)+'_c'+str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir, stimulustype+'_images/')
    Results_GT_dir = os.path.join(Results_base_dir, stimulustype+'_GT/')
    Results_GT_bp_dir = os.path.join(Results_base_dir, stimulustype+'_GT_barberpole/')
    for d in [Results_dir, Results_GT_dir, Results_GT_bp_dir]:
        if not os.path.exists(os.path.dirname(d)):
            os.makedirs(os.path.dirname(d))

    for stimfreqies in itertools.combinations(frequencies, 2):
        for orientations_iter in itertools.product(orientations, orientation_offset):
            current_orientations = np.asarray(orientations_iter)
            current_orientations[1] = current_orientations[1] + current_orientations[0]
            vel_sample = np.logspace(0, np.log10(0.5/np.max((stimfreqies[0], stimfreqies[1]))), 8, endpoint=True)

            for current_speeds in itertools.combinations(vel_sample, 2):
                for aper_orient in aperture_orientations:
                    a = GeneratePlaid(Height, Width, stimfreqies[0], current_orientations[0], current_speeds[0],
                                      stimfreqies[1], current_orientations[1], current_speeds[1])
                    for index, dispfactor in enumerate([-2, -1, 0, 1, 2]):
                        current_frame = np.zeros([Height, Width], dtype=np.float64)
                        flow_gt_ioc = np.zeros([2, Height, Width], dtype=np.float64)
                        flow_gt_bp = np.zeros([2, Height, Width], dtype=np.float64)

                        # Rectangular aperture
                        half_l = rect_length / 2.0
                        half_w = rect_width / 2.0
                        cos_a = np.cos(np.deg2rad(-aper_orient))
                        sin_a = np.sin(np.deg2rad(-aper_orient))
                        cy, cx = Height//2 - 1, Width//2 - 1
                        corners_x = np.array([-half_l, half_l, half_l, -half_l])
                        corners_y = np.array([-half_w, -half_w, half_w, half_w])
                        rot_x = corners_x * cos_a - corners_y * sin_a + cx
                        rot_y = corners_x * sin_a + corners_y * cos_a + cy
                        mask = np.zeros([Height, Width], dtype=np.uint8)
                        pts = np.array(list(zip(rot_x.astype(int), rot_y.astype(int))), dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                        tyc, txc = np.where(mask > 0)

                        current_frame[tyc, txc] = contrastvalue * a[tyc, txc, index]

                        # IoC velocity
                        if current_speeds[0] == current_speeds[1]:
                            ioc_vx = 0.5*((current_speeds[0]*np.cos(np.deg2rad(current_orientations[0]))) + (current_speeds[1]*np.cos(np.deg2rad(current_orientations[1]))))
                            ioc_vy = -1*0.5*((current_speeds[0]*np.sin(np.deg2rad(current_orientations[0]))) + (current_speeds[1]*np.sin(np.deg2rad(current_orientations[1]))))
                        else:
                            denom = np.sin(np.deg2rad(current_orientations[1]) - np.deg2rad(current_orientations[0]))
                            ioc_vx = (current_speeds[0]*np.sin(np.deg2rad(current_orientations[1])) - current_speeds[1]*np.sin(np.deg2rad(current_orientations[0]))) / denom
                            ioc_vy = -1*(current_speeds[0]*np.cos(np.deg2rad(current_orientations[1])) - current_speeds[1]*np.cos(np.deg2rad(current_orientations[0]))) / np.sin(np.deg2rad(current_orientations[0]) - np.deg2rad(current_orientations[1]))

                        flow_gt_ioc[0, tyc, txc] = ioc_vx
                        flow_gt_ioc[1, tyc, txc] = ioc_vy

                        # Barber pole: project IoC velocity onto aperture long axis
                        ax_x = np.cos(np.deg2rad(aper_orient))
                        ax_y = -np.sin(np.deg2rad(aper_orient))
                        proj = ioc_vx * ax_x + ioc_vy * ax_y
                        flow_gt_bp[0, tyc, txc] = proj * ax_x
                        flow_gt_bp[1, tyc, txc] = proj * ax_y

                        VX = np.round(ioc_vx)
                        VY = np.round(ioc_vy)
                        I_flow_ioc = viz_flow(flow_gt_ioc[0,:,:], flow_gt_ioc[1,:,:])
                        I_flow_bp = viz_flow(flow_gt_bp[0,:,:], flow_gt_bp[1,:,:])

                        name_base = stimulustype + '_cyc_'+str(np.round(stimfreqies[0],4))+'_'+str(np.round(stimfreqies[1],4)) + \
                            '_orient_'+str(current_orientations[0])+'_'+str(current_orientations[1]) + \
                            '_aorient_'+str(aper_orient) + \
                            '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)
                        img_name = os.path.join(Results_dir, name_base+'.png')
                        gt_ioc_flo = os.path.join(Results_GT_dir, name_base+'.flo')
                        gt_ioc_vis = os.path.join(Results_GT_dir, name_base+'.png')
                        gt_bp_flo = os.path.join(Results_GT_bp_dir, name_base+'.flo')
                        gt_bp_vis = os.path.join(Results_GT_bp_dir, name_base+'.png')

                        cv2.imwrite(img_name, cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                        if index < 4:
                            cv2.imwrite(gt_ioc_vis, cv2.cvtColor(I_flow_ioc, cv2.COLOR_RGB2BGR))
                            flow_write(gt_ioc_flo, flow_gt_ioc[0,:,:], flow_gt_ioc[1,:,:])
                            cv2.imwrite(gt_bp_vis, cv2.cvtColor(I_flow_bp, cv2.COLOR_RGB2BGR))
                            flow_write(gt_bp_flo, flow_gt_bp[0,:,:], flow_gt_bp[1,:,:])


def GenerateTransparentMotion(Height, Width, num_dots, dot_radius, speed_pairs, direction_pairs, Results_base_dir):
    """Generate transparent motion stimuli: two overlapping dot surfaces moving in different directions.

    Two populations of random dots occupy the same spatial region, each moving coherently
    in a different direction. The visual system segments them into two transparent surfaces.

    Ground truth provides two flow layers:
      - Layer 0: flow for surface 1
      - Layer 1: flow for surface 2
    Plus a surface-membership mask indicating which dots belong to which surface.
    """
    stimulustype = 'transparent_ndots_'+str(num_dots)+'_r'+str(dot_radius)
    Results_dir = os.path.join(Results_base_dir, stimulustype+'_images/')
    Results_GT_s1_dir = os.path.join(Results_base_dir, stimulustype+'_GT_surface1/')
    Results_GT_s2_dir = os.path.join(Results_base_dir, stimulustype+'_GT_surface2/')
    Results_mask_dir = os.path.join(Results_base_dir, stimulustype+'_masks/')
    for d in [Results_dir, Results_GT_s1_dir, Results_GT_s2_dir, Results_mask_dir]:
        if not os.path.exists(os.path.dirname(d)):
            os.makedirs(os.path.dirname(d))

    np.random.seed(42)
    margin = 80  # keep dots away from border to avoid clipping at max displacement
    contrastvalue = 200

    for speed1, speed2 in speed_pairs:
        for dir1, dir2 in direction_pairs:
            vx1 = speed1 * np.cos(np.deg2rad(dir1))
            vy1 = -speed1 * np.sin(np.deg2rad(dir1))
            vx2 = speed2 * np.cos(np.deg2rad(dir2))
            vy2 = -speed2 * np.sin(np.deg2rad(dir2))

            # Generate random dot positions (shared across temporal frames via displacement)
            half_dots = num_dots // 2
            # Surface 1 dot centers
            cx_s1 = np.random.randint(margin, Width - margin, half_dots)
            cy_s1 = np.random.randint(margin, Height - margin, half_dots)
            # Surface 2 dot centers
            cx_s2 = np.random.randint(margin, Width - margin, half_dots)
            cy_s2 = np.random.randint(margin, Height - margin, half_dots)

            for index, dispfactor in enumerate([-2, -1, 0, 1, 2]):
                current_frame = np.zeros([Height, Width], dtype=np.float64)
                flow_gt_s1 = np.zeros([2, Height, Width], dtype=np.float64)
                flow_gt_s2 = np.zeros([2, Height, Width], dtype=np.float64)
                surface_mask = np.zeros([Height, Width], dtype=np.uint8)  # 1=surface1, 2=surface2

                # Draw surface 1 dots
                for i in range(half_dots):
                    dx = int(round(vx1 * dispfactor))
                    dy = int(round(vy1 * dispfactor))
                    x = cx_s1[i] + dx
                    y = cy_s1[i] + dy
                    if 0 <= x < Width and 0 <= y < Height:
                        cv2.circle(current_frame, (x, y), dot_radius, contrastvalue, -1)
                        # Mark flow for this dot's pixels
                        dot_mask = np.zeros([Height, Width], dtype=np.uint8)
                        cv2.circle(dot_mask, (x, y), dot_radius, 255, -1)
                        dy_px, dx_px = np.where(dot_mask > 0)
                        flow_gt_s1[0, dy_px, dx_px] = vx1
                        flow_gt_s1[1, dy_px, dx_px] = vy1
                        surface_mask[dy_px, dx_px] = 1

                # Draw surface 2 dots (with different intensity for visualization)
                for i in range(half_dots):
                    dx = int(round(vx2 * dispfactor))
                    dy = int(round(vy2 * dispfactor))
                    x = cx_s2[i] + dx
                    y = cy_s2[i] + dy
                    if 0 <= x < Width and 0 <= y < Height:
                        cv2.circle(current_frame, (x, y), dot_radius, contrastvalue * 0.6, -1)
                        dot_mask = np.zeros([Height, Width], dtype=np.uint8)
                        cv2.circle(dot_mask, (x, y), dot_radius, 255, -1)
                        dy_px, dx_px = np.where(dot_mask > 0)
                        flow_gt_s2[0, dy_px, dx_px] = vx2
                        flow_gt_s2[1, dy_px, dx_px] = vy2
                        surface_mask[dy_px, dx_px] = 2

                VX1 = np.round(vx1)
                VY1 = np.round(vy1)
                VX2 = np.round(vx2)
                VY2 = np.round(vy2)

                I_flow_s1 = viz_flow(flow_gt_s1[0,:,:], flow_gt_s1[1,:,:])
                I_flow_s2 = viz_flow(flow_gt_s2[0,:,:], flow_gt_s2[1,:,:])

                name_base = stimulustype + \
                    '_dir1_'+str(dir1)+'_spd1_'+str(speed1) + \
                    '_dir2_'+str(dir2)+'_spd2_'+str(speed2) + \
                    '__'+str(index).zfill(2)
                img_name = os.path.join(Results_dir, name_base+'.png')
                gt_s1_flo = os.path.join(Results_GT_s1_dir, name_base+'.flo')
                gt_s1_vis = os.path.join(Results_GT_s1_dir, name_base+'.png')
                gt_s2_flo = os.path.join(Results_GT_s2_dir, name_base+'.flo')
                gt_s2_vis = os.path.join(Results_GT_s2_dir, name_base+'.png')
                mask_name = os.path.join(Results_mask_dir, name_base+'.png')

                cv2.imwrite(img_name, cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                cv2.imwrite(mask_name, surface_mask * 127)  # 0=bg, 127=s1, 254=s2
                if index < 4:
                    cv2.imwrite(gt_s1_vis, cv2.cvtColor(I_flow_s1, cv2.COLOR_RGB2BGR))
                    flow_write(gt_s1_flo, flow_gt_s1[0,:,:], flow_gt_s1[1,:,:])
                    cv2.imwrite(gt_s2_vis, cv2.cvtColor(I_flow_s2, cv2.COLOR_RGB2BGR))
                    flow_write(gt_s2_flo, flow_gt_s2[0,:,:], flow_gt_s2[1,:,:])


def GenerateTransparentGratings(Height, Width, Radius, frequencies, orientations, orientation_offset, Results_base_dir):
    """Generate transparent motion with overlapping gratings moving in different directions.

    Two gratings at different orientations/speeds are superimposed within a circular aperture,
    but unlike plaids (where IoC yields a single coherent velocity), here the ground truth
    preserves each grating's component velocity as a separate layer.

    This tests whether a method can decompose the motion into two transparent surfaces
    rather than computing a single averaged/IoC velocity.
    """
    contrastvalue = np.random.randint(10, 255, 1)[0]
    stimulustype = 'transparent_gratings_R_'+str(Radius)+'_c'+str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir, stimulustype+'_images/')
    Results_GT_s1_dir = os.path.join(Results_base_dir, stimulustype+'_GT_surface1/')
    Results_GT_s2_dir = os.path.join(Results_base_dir, stimulustype+'_GT_surface2/')
    Results_GT_ioc_dir = os.path.join(Results_base_dir, stimulustype+'_GT_ioc/')
    for d in [Results_dir, Results_GT_s1_dir, Results_GT_s2_dir, Results_GT_ioc_dir]:
        if not os.path.exists(os.path.dirname(d)):
            os.makedirs(os.path.dirname(d))

    for stimfreqies in itertools.combinations(frequencies, 2):
        for orientations_iter in itertools.product(orientations, orientation_offset):
            current_orientations = np.asarray(orientations_iter)
            current_orientations[1] = current_orientations[1] + current_orientations[0]
            vel_sample = np.logspace(0, np.log10(0.5/np.max((stimfreqies[0], stimfreqies[1]))), 8, endpoint=True)

            for current_speeds in itertools.combinations(vel_sample, 2):
                g1 = GenerateGrating(Height, Width, stimfreqies[0], current_orientations[0], current_speeds[0])
                g2 = GenerateGrating(Height, Width, stimfreqies[1], current_orientations[1], current_speeds[1])

                for index, dispfactor in enumerate([-2, -1, 0, 1, 2]):
                    current_frame = np.zeros([Height, Width], dtype=np.float64)
                    flow_gt_s1 = np.zeros([2, Height, Width], dtype=np.float64)
                    flow_gt_s2 = np.zeros([2, Height, Width], dtype=np.float64)
                    flow_gt_ioc = np.zeros([2, Height, Width], dtype=np.float64)

                    tyc, txc = draw.circle(Height//2-1, Width//2-1, radius=Radius, shape=current_frame.shape)
                    current_frame[tyc, txc] = contrastvalue * (0.5*g1[tyc, txc, index] + 0.5*g2[tyc, txc, index])

                    # Surface 1: component velocity of grating 1
                    flow_gt_s1[0, tyc, txc] = current_speeds[0] * np.cos(np.deg2rad(current_orientations[0]))
                    flow_gt_s1[1, tyc, txc] = -1 * current_speeds[0] * np.sin(np.deg2rad(current_orientations[0]))

                    # Surface 2: component velocity of grating 2
                    flow_gt_s2[0, tyc, txc] = current_speeds[1] * np.cos(np.deg2rad(current_orientations[1]))
                    flow_gt_s2[1, tyc, txc] = -1 * current_speeds[1] * np.sin(np.deg2rad(current_orientations[1]))

                    # IoC velocity (for comparison)
                    if current_speeds[0] == current_speeds[1]:
                        flow_gt_ioc[0, tyc, txc] = 0.5*((current_speeds[0]*np.cos(np.deg2rad(current_orientations[0]))) + (current_speeds[1]*np.cos(np.deg2rad(current_orientations[1]))))
                        flow_gt_ioc[1, tyc, txc] = -1*0.5*((current_speeds[0]*np.sin(np.deg2rad(current_orientations[0]))) + (current_speeds[1]*np.sin(np.deg2rad(current_orientations[1]))))
                    else:
                        denom = np.sin(np.deg2rad(current_orientations[1]) - np.deg2rad(current_orientations[0]))
                        flow_gt_ioc[0, tyc, txc] = (current_speeds[0]*np.sin(np.deg2rad(current_orientations[1])) - current_speeds[1]*np.sin(np.deg2rad(current_orientations[0]))) / denom
                        flow_gt_ioc[1, tyc, txc] = -1*(current_speeds[0]*np.cos(np.deg2rad(current_orientations[1])) - current_speeds[1]*np.cos(np.deg2rad(current_orientations[0]))) / np.sin(np.deg2rad(current_orientations[0]) - np.deg2rad(current_orientations[1]))

                    I_flow_s1 = viz_flow(flow_gt_s1[0,:,:], flow_gt_s1[1,:,:])
                    I_flow_s2 = viz_flow(flow_gt_s2[0,:,:], flow_gt_s2[1,:,:])
                    I_flow_ioc = viz_flow(flow_gt_ioc[0,:,:], flow_gt_ioc[1,:,:])

                    VX_ioc = np.round(flow_gt_ioc[0, tyc[0], txc[0]])
                    VY_ioc = np.round(flow_gt_ioc[1, tyc[0], txc[0]])
                    name_base = stimulustype + '_cyc_'+str(np.round(stimfreqies[0],4))+'_'+str(np.round(stimfreqies[1],4)) + \
                        '_orient_'+str(current_orientations[0])+'_'+str(current_orientations[1]) + \
                        '_VX_'+str(VX_ioc).zfill(4)+'_VY_'+str(VY_ioc).zfill(4)+'__'+str(index).zfill(2)

                    img_name = os.path.join(Results_dir, name_base+'.png')
                    cv2.imwrite(img_name, cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                    if index < 4:
                        for gt_dir, flow_arr, flow_vis in [
                            (Results_GT_s1_dir, flow_gt_s1, I_flow_s1),
                            (Results_GT_s2_dir, flow_gt_s2, I_flow_s2),
                            (Results_GT_ioc_dir, flow_gt_ioc, I_flow_ioc)]:
                            cv2.imwrite(os.path.join(gt_dir, name_base+'.png'), cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR))
                            flow_write(os.path.join(gt_dir, name_base+'.flo'), flow_arr[0,:,:], flow_arr[1,:,:])


def GenerateMovingGratings(Height,Width,Radius, orientations, Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    #contrastvalue =  255;
    contrastvalue =  np.random.randint(10,255,1)[0];
    stimulustype = 'gratings_circ_aper_R_'+str(Radius)+'_c'+ str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/') 
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    if not os.path.exists(os.path.dirname(Results_GT_dir)):
        os.makedirs(os.path.dirname(Results_GT_dir))
    frequencies = np.divide(1.0,range(2,2*Radius,8))
    for stimfreq in frequencies:
        for current_orientation in orientations:
            vel_tsample = np.logspace(0, np.log10(0.5/stimfreq), 8, endpoint=True)  # v = ft x lambda -> ft = v/lambda -> ft = v x fspatial  
            vel_sample = (np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0))) # We are taking 5 temporal samples, so frequency should be less than 2 to avoid aliasing
            tic()            
            for current_speed in vel_sample:
                #Generate a ciruclar aperture of the radius
                #Generate gradings under Square shape
                #Put the grating stimuli within circularshape
                a = GenerateGrating(Height,Width, stimfreq,(current_orientation),current_speed)
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width],dtype=np.float)
                    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                    tyc,txc = draw.circle( Height/2-1,Width/2-1, radius=Radius, shape=current_frame.shape)
                    current_frame[tyc,txc] = contrastvalue*a[tyc,txc,index]
                    flow_gt_current[0,tyc,txc] = current_speed*np.cos(np.deg2rad(current_orientation)) #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc,txc] = -1*current_speed*np.sin(np.deg2rad(current_orientation)) # flipping coordinate axis to match 
                    VX = np.round(current_speed*np.cos(np.deg2rad(current_orientation)))
                    VY = -1*np.round(current_speed*np.sin(np.deg2rad(current_orientation)))
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #plt.imshow(I_flow)
                    #plt.show()
                    #pdb.set_trace()
                    ##
                    #Writing files to disk
                    img_name = os.path.join(Results_dir,stimulustype + '_cyc_'+str(np.round(2*Radius*stimfreq))+'_orient_'+ str(current_orientation)+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_GT_dir,stimulustype+ '_cyc_'+str(np.round(2*Radius*stimfreq)) + '_orient_'+ str(current_orientation)  +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = os.path.join(Results_GT_dir,stimulustype + '_cyc_'+str(np.round(2*Radius*stimfreq))+ '_orient_'+ str(current_orientation) +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    #cv2.imwrite(img_name,current_frame)
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
            toc()



# if __name__ == '__main__':
#     Width = 1024 # 1280
#     Height = 436    # 768
#     vel_tsample = np.logspace(0, np.log10(25), 10, endpoint=True)
#     #Results_base_dir = '/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
#     #Results_base_dir = '/home/Work/Code/StimuliGenerators/Stimuli_Samples/'
#     #Results_base_dir = '/home/MedathatiExt/OpticalFlow/Stimuli/Stimuli_Samples/'
#     #Results_base_dir = '/run/media/medathati/MedathatiExt/OpticalFlow/Stimuli/Stimuli_Samples/'
#     #Results_base_dir = '/run/media/medathati/MedathatiExt/OpticalFlow/Stimuli/Wallach_dataset/'
#     #Results_base_dir = '/run/media/medathati/MedathatiExt/OpticalFlow/Stimuli/Wallach_dataset/'
#     Results_base_dir = '/run/media/medathati/4TBInt/MedathatiExt/Work/OpticalFlow/Data/Stimuli/April/'
#     circ_radii = np.array([21,42])
    
#     for circ_radius in circ_radii:
#         print("Generating moving circles with ",circ_radius)
#         GenerateMovingCircle(Height, Width,circ_radius,'perimeter', vel_tsample,Results_base_dir)
#         GenerateMovingCircle(Height, Width,circ_radius,'full', vel_tsample,Results_base_dir)
    
#     orientations = np.linspace(0,180,7)
#     orientations = orientations[:-1]
#     #Aperture_Radius = 100
#     #GenerateMovingPlaids(Height,Width,Aperture_Radius, orientations, Results_base_dir)
#     Aperture_Radii  = np.array([50,75,100,125])
#     for Aperture_Radius in Aperture_Radii:
#         print("Generating moving gratings with ",Aperture_Radius)
#         GenerateMovingGratings(Height,Width,Aperture_Radius, orientations, Results_base_dir) 
    
#     line_lengths = np.array([10,20,30])
#     for line_length in line_lengths:
#         print("Generating moving lines with ",line_lengths)
#         GenerateMovingLine(Height, Width,line_length,orientations, vel_tsample,Results_base_dir)
#     Aperture_Radius = 100
#     GenerateMovingPlaids(Height,Width,Aperture_Radius, orientations, Results_base_dir) 
#     VisualizeVelocitySamples(Height, Width,vel_tsample) 
#     #GenerateMovingRectangles(Height, Width,50,0.75,orientations, vel_tsample,Results_base_dir)

