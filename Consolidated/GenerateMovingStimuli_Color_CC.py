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
    print fmt % (time() - _tstart_stack.pop())
    

def VisualizeVelocitySamples(Height, Width,vel_tsample):
    xc = Width/2 -1
    yc = Height/2-1
    contrastvalue =  255;
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    current_frame = np.zeros([Height, Width],dtype=np.float)
    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
    for VY in vel_sample:
        for VX in vel_sample:
            yc,xc = draw.circle_perimeter( Height/2-1,Width/2-1, radius=3, shape=current_frame.shape)
            txc = xc + VX
            tyc=  yc - VY
            current_frame[tyc,txc]= contrastvalue
            flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
            flow_gt_current[1,tyc,txc] = -VY #
    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
    
    #plt.imshow(I_flow)
    #plt.show()
    ##plt.quiver(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
    #plt.show()
    ##
    #Writing files to disk
    cv2.imwrite('VelocitySampling.png',current_frame)
    cv2.imwrite('VelocitySampling_colorcode.png', cv2.cvtColor(I_flow, cv2.COLOR_RGB2BGR))
    
                
def GenerateMovingCircle(Height, Width,circle_radius,circ_type, vel_tsample,Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    stimulustype = 'moving_circle_'+circ_type+'_r'+str(circle_radius)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/')  #'/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    #if not os.path.exists(os.path.dirname(Results_GT_dir)):
    #    os.makedirs(os.path.dirname(Results_GT_dir))
    
    #vel_tsample = np.logspace(0, np.log10(100), 10, endpoint=True)
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    for VY in vel_sample: #range(-50,51):#range(-(Height/4)+1,Height/4):
        for VX in vel_sample: #range(-50,51):#range(-(Width/4)+1,Width/4):#tic()
	    contrastvalue =  np.random.randint(1,255,size=3);
            for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                img_name = os.path.join(Results_dir,stimulustype+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                gt_flo_name = os.path.join(Results_dir,stimulustype+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                gt_flo_name_vis = os.path.join(Results_dir,stimulustype+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'_GT.png')
                #Creating the buffer for the frame and flow ground truth
                current_frame = np.zeros([Height, Width,3],dtype=np.float)
                flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                #Computing the displacement for the pixels
                if circ_type =='perimeter':
                    yc,xc = draw.circle_perimeter( Height/2-1,Width/2-1, radius=circle_radius, shape=current_frame.shape) #-1 since coordinate system begins at zero
                if circ_type =='full':
                    yc,xc = draw.circle( Height/2-1,Width/2-1, radius=circle_radius, shape=current_frame.shape) #-1 since coordinate system begins at zero
                disp_X = VX*dispfactor
                disp_Y = VY*dispfactor
                txc = xc + disp_X
                tyc=  yc - disp_Y
                current_frame[tyc,txc,0] = contrastvalue[0];
                current_frame[tyc,txc,1] = contrastvalue[1];
                current_frame[tyc,txc,2] = contrastvalue[2];
                #Setting ground truth velocity
                flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                flow_gt_current[1,tyc,txc] = VY #
                I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
		#print(np.shape(I_flow))
                #plt.imshow(I_flow)
                #plt.show()
                ##
                #Writing files to disk
                #cv2.imwrite(img_name,current_frame)
                cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_RGB2BGR))
		if index<4:
	                cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
        	        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
            #toc()
            #print VX, VY




def GenerateMovingLine(Height, Width,length,orientations, vel_tsample,Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    stimulustype = 'moving_line_l_'+str(length)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/')  #'/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    #if not os.path.exists(os.path.dirname(Results_GT_dir)):
    #    os.makedirs(os.path.dirname(Results_GT_dir))
    
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    for current_orientation in orientations:
        for VY in vel_sample: #range(-50,51):#range(-(Height/4)+1,Height/4):
            for VX in vel_sample: #range(-50,51):#range(-(Width/4)+1,Width/4):
		contrastvalue =  np.random.randint(1,255,3)
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    img_name = os.path.join(Results_dir,stimulustype + '_orient_'+ str(current_orientation)+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_dir,stimulustype + '_orient_'+ str(current_orientation)  +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = os.path.join(Results_dir,stimulustype + '_orient_'+ str(current_orientation) +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'_GT.png')
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width,3],dtype=np.float)
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
                    tyc=  yc - disp_Y
                    current_frame[tyc,txc,0] = contrastvalue[0];
                    current_frame[tyc,txc,1] = contrastvalue[1];
                    current_frame[tyc,txc,2] = contrastvalue[2];
                    #Setting ground truth velocity
                    flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc,txc] = VY #
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #plt.imshow(I_flow)
                    #plt.show()
                    ##
                    #Writing files to disk
                    #cv2.imwrite(img_name,current_frame)
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_RGB2BGR))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])


def GenerateMovingLine_undercircaper(Height, Width,length,circle_radius,orientations, vel_tsample,Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    contrastvalue =  np.random.randint(1,255,1);
    stimulustype = 'moving_line_l__undercircaper_'+str(length)+'_c'+ str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/')  #'/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    #if not os.path.exists(os.path.dirname(Results_GT_dir)):
    #    os.makedirs(os.path.dirname(Results_GT_dir))
    
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
                    tyc=  yc - disp_Y
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

def GenerateMovingRectangles(Height, Width,length,aspectratio,orientations, vel_tsample,Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    stimulustype = 'moving_rectangle_l_'+str(length)+'_aspr_'+str(aspectratio)+'_c'+ str(contrastvalue)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/')  #'/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    #if not os.path.exists(os.path.dirname(Results_GT_dir)):
    #    os.makedirs(os.path.dirname(Results_GT_dir))
    
    #vel_tsample = np.logspace(0, np.log10(100), 10, endpoint=True)
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    for current_orientation in orientations:
        for VY in vel_sample: #range(-50,51):#range(-(Height/4)+1,Height/4):
            for VX in vel_sample: #range(-50,51):#range(-(Width/4)+1,Width/4):
		contrastvalue =  np.random.randint(10,255,3);
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    img_name = os.path.join(Results_dir,stimulustype + '_orient_'+ str(current_orientation)+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_GT_dir,stimulustype + '_orient_'+ str(current_orientation)  +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = os.path.join(Results_GT_dir,stimulustype + '_orient_'+ str(current_orientation) +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width,3],dtype=np.float)
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
                    tyc=  yc - disp_Y
                    current_frame[tyc,txc,0] = contrastvalue[0];
                    current_frame[tyc,txc,1] = contrastvalue[1];
                    current_frame[tyc,txc,2] = contrastvalue[2];
                    #Setting ground truth velocity
                    flow_gt_current[0,tyc,txc] = VX #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc,txc] = VY #
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #plt.imshow(I_flow)
                    #plt.show()
                    ##
                    #Writing files to disk
                    #cv2.imwrite(img_name,current_frame)
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_RGB2BGR))
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
    
def GenerateMovingPlaids(Height,Width,Radius, orientations, Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    stimulustype = 'moving_plaids_circ_aper_R_'+str(Radius)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/') 
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    #if not os.path.exists(os.path.dirname(Results_GT_dir)):
    #    os.makedirs(os.path.dirname(Results_GT_dir))
    #frequencies = np.divide(1.0,range(2,2*Radius,8))
    #frequencies = np.divide(1.0,range(2,2*Radius,8))
    frequencies = [1.0/10,1.0/10]
    orientation_offset = np.linspace(30.0,90.0,5)
    orientation_offset = orientation_offset[np.nonzero(orientation_offset)]
    for stimfreqies in itertools.combinations(frequencies,2):
        for orientations_iter in itertools.product(orientations,orientation_offset):
	    current_orientations = np.asarray(orientations_iter)
            current_orientations[1] = current_orientations[1]+current_orientations[0]
            vel_sample = np.logspace(0, np.log10(0.5/np.max((stimfreqies[0],stimfreqies[1]))), 8, endpoint=True)   
            #vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
            tic()            
            for current_speeds in itertools.combinations(vel_sample,2):
		contrastvalue =  np.random.randint(10,255,3);
                a = GeneratePlaid(Height,Width, stimfreqies[0],current_orientations[0],current_speeds[0],stimfreqies[1],current_orientations[1], current_speeds[1])
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width,3],dtype=np.float)
                    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                    tyc,txc = draw.circle( Height/2-1,Width/2-1, radius=Radius, shape=current_frame.shape)
                    current_frame[tyc,txc,0] = contrastvalue[0]*a[tyc,txc,index]
                    current_frame[tyc,txc,1] = contrastvalue[1]*a[tyc,txc,index]
                    current_frame[tyc,txc,2] = contrastvalue[2]*a[tyc,txc,index]
                    ## Compute IoC
                     
                    flow_gt_current[0,tyc,txc] = (current_speeds[0]*np.sin(np.deg2rad(current_orientations[0]))-current_speeds[1]*np.sin(np.deg2rad(current_orientations[1])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))
                        
                    flow_gt_current[1,tyc,txc] = (current_speeds[0]*np.cos(np.deg2rad(current_orientations[0]))-current_speeds[1]*np.cos(np.deg2rad(current_orientations[1])))/np.sin(np.deg2rad(current_orientations[1])-np.deg2rad(current_orientations[0]))
                    #flow_gt_current[0,tyc,txc] = current_speed*np.cos(np.deg2rad(current_orientation)) #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    #flow_gt_current[1,tyc,txc] = current_speed*np.sin(np.deg2rad(current_orientation)) #
                    VX = np.round(flow_gt_current[0,tyc[0],txc[0]])
                    VY = np.round(flow_gt_current[1,tyc[0],txc[0]])
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #Writing files to disk
                    img_name = os.path.join(Results_dir,stimulustype + '_cyc_'+str(np.round(2*Radius*stimfreqies[0]))+ '_' + str(np.round(2*Radius*stimfreqies[1])) + \
                        '_orient_' + str(current_orientations[0]) + '_' + str(current_orientations[1]) + \
                        '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__' + str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_dir,stimulustype+ '_cyc_'+str(np.round(2*Radius*stimfreqies[0])) + '_' + str(np.round(2*Radius*stimfreqies[1])) + \
                        '_orient_' + str(current_orientations[0]) + '_' + str(current_orientations[1]) + \
                        '_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = gt_flo_name[:-4]+'_GT.png'
                    #cv2.imwrite(img_name,current_frame)
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_RGB2BGR))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
            toc

def GenerateMovingGratings(Height,Width,Radius, orientations, Results_base_dir):
    xc = Width/2 -1
    yc = Height/2-1
    #contrastvalue =  255;
    stimulustype = 'moving_gratings_circ_aper_R_'+str(Radius)
    Results_dir = os.path.join(Results_base_dir,stimulustype+'_images/') 
    Results_GT_dir = os.path.join(Results_base_dir,stimulustype+'_GT/')
    if not os.path.exists(os.path.dirname(Results_dir)):
        os.makedirs(os.path.dirname(Results_dir))
    #if not os.path.exists(os.path.dirname(Results_GT_dir)):
    #    os.makedirs(os.path.dirname(Results_GT_dir))
    frequencies = np.divide(1.0,range(2,2*Radius,8))
    for stimfreq in frequencies:
        for current_orientation in orientations:
            vel_tsample = np.logspace(0, np.log10(0.5/stimfreq), 8, endpoint=True)  # v = ft x lambda -> ft = v/lambda -> ft = v x fspatial  
            vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int) # We are taking 5 temporal samples, so frequency should be less than 2 to avoid aliasing
            #vel_sample = np.round(np.concatenate(([0],vel_tsample),axis=0)).astype(int) # We are taking 5 temporal samples, so frequency should be less than 2 to avoid aliasing
            tic()            
            for current_speed in vel_sample:
                #Generate a ciruclar aperture of the radius
                #Generate gradings under Square shape
                #Put the grating stimuli within circularshape
		contrastvalue =  np.random.randint(10,255,3);
                a = GenerateGrating(Height,Width, stimfreq,(current_orientation),current_speed)
                for index, dispfactor in enumerate([-2, -1, 0,1,2]):
                    #Creating the buffer for the frame and flow ground truth
                    current_frame = np.zeros([Height, Width,3],dtype=np.float)
                    flow_gt_current = np.zeros([2,Height, Width],dtype=np.float)
                    tyc,txc = draw.circle( Height/2-1,Width/2-1, radius=Radius, shape=current_frame.shape)
                    current_frame[tyc,txc,0] = contrastvalue[0]*a[tyc,txc,index]
                    current_frame[tyc,txc,1] = contrastvalue[1]*a[tyc,txc,index]
                    current_frame[tyc,txc,2] = contrastvalue[2]*a[tyc,txc,index]
                    flow_gt_current[0,tyc,txc] = current_speed*np.cos(np.deg2rad(current_orientation)) #   tyc-3:tyc+3,txc-3:txc+3 tyc,txc
                    flow_gt_current[1,tyc,txc] = current_speed*np.sin(np.deg2rad(current_orientation)) #
                    VX = np.round(current_speed*np.cos(np.deg2rad(current_orientation)))
                    VY = np.round(current_speed*np.sin(np.deg2rad(current_orientation)))
                    I_flow = viz_flow(flow_gt_current[0,:,:], flow_gt_current[1,:,:])
                    #plt.imshow(I_flow)
                    #plt.show()
                    #pdb.set_trace()
                    ##
                    #Writing files to disk
                    img_name = os.path.join(Results_dir,stimulustype + '_cyc_'+str(np.round(2*Radius*stimfreq))+'_orient_'+ str(current_orientation)+'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.png')
                    gt_flo_name = os.path.join(Results_dir,stimulustype+ '_cyc_'+str(np.round(2*Radius*stimfreq)) + '_orient_'+ str(current_orientation)  +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'.flo')
                    gt_flo_name_vis = os.path.join(Results_dir,stimulustype + '_cyc_'+str(np.round(2*Radius*stimfreq))+ '_orient_'+ str(current_orientation) +'_VX_'+str(VX).zfill(4)+'_VY_'+str(VY).zfill(4)+'__'+str(index).zfill(2)+'_GT.png')
                    #cv2.imwrite(img_name,current_frame)
                    cv2.imwrite(img_name,cv2.cvtColor(current_frame.astype('uint8'), cv2.COLOR_RGB2BGR))
                    if index<4:
                        cv2.imwrite(gt_flo_name_vis, cv2.cvtColor( I_flow, cv2.COLOR_RGB2BGR))
                        flow_write(gt_flo_name, flow_gt_current[0,:,:], flow_gt_current[1,:,:])
            toc()



if __name__ == '__main__':
    Width = 1024 # 1280
    Height = 436    # 768
    vel_tsample = np.logspace(0, np.log10(25), 10, endpoint=True)
    VisualizeVelocitySamples(Height, Width,vel_tsample) 
    #Results_base_dir = '/home/medathati/Work/OpticalFlow/Code/WithFabio/StimuliGenerators/Results/Results_WDrive/'
    #Results_base_dir = '/home/Work/Code/StimuliGenerators/Stimuli_Samples/'
    #Results_base_dir = '/home/MedathatiExt/OpticalFlow/Stimuli/Stimuli_Samples/'
    #Results_base_dir = '/run/media/medathati/MedathatiExt/OpticalFlow/Stimuli/Stimuli_Samples/'
    #Results_base_dir = '/run/media/medathati/MedathatiExt/OpticalFlow/Stimuli/ColorVersion/'
    Results_base_dir = '/home/SDD_Work/ColorVersion_CC1/'
    
    '''
    circ_radii = np.array([21,42])
    for circ_radius in circ_radii:
        print("Generating moving circles with ",circ_radius)
	tic()
        GenerateMovingCircle(Height, Width,circ_radius,'perimeter', vel_tsample,Results_base_dir)
        GenerateMovingCircle(Height, Width,circ_radius,'full', vel_tsample,Results_base_dir)
	toc()
    
    orientations = np.linspace(0,0,13)
    orientations = orientations[:-1]
    Aperture_Radii  = np.array([50,75,100])
    for Aperture_Radius in Aperture_Radii:
        print("Generating moving gratings with ",Aperture_Radius)
	tic()
        GenerateMovingGratings(Height,Width,Aperture_Radius, orientations, Results_base_dir) 
	toc()
    
    line_lengths = np.array([30,100])
    orientations = np.linspace(0,180,7)
    orientations = orientations[:-1] 
    for line_length in line_lengths:
        print("Generating moving lines with ",line_lengths)
	tic()
        GenerateMovingLine(Height, Width,line_length,orientations, vel_tsample,Results_base_dir)
	toc() 
    tic()
    '''
    #orientations = np.linspace(0,360,13)
    orientations = np.linspace(0,270,10)
    #orientations = orientations[:-1] 
    Aperture_Radius = 100
    print("Generating moving plaids, aperture: ",Aperture_Radius)
    GenerateMovingPlaids(Height,Width,Aperture_Radius, orientations, Results_base_dir) 
    toc()   
    #GenerateMovingRectangles(Height, Width,50,0.75,orientations, vel_tsample,Results_base_dir)

