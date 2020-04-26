#!/usr/bin/env python

import torch
import torch.utils.data as data

import getopt
import math
import numpy
import numpy as np
import os
import os, math, random
import PIL
import PIL.Image
import sys

from os.path import *
from glob import glob
from scipy.misc import imread, imresize

from flowlib import *
from f2i import Flow 
import matplotlib
import matplotlib.pylab as plt
plt.switch_backend('pdf')
#matplotlib.use('pdf')
##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'sintel-final'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, see below
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super(Preprocess, self).__init__()
			# end

			def forward(self, tensorInput):
				tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
				tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
				tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

				return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)
			# end
		# end

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super(Basic, self).__init__()

				self.moduleBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleBasic(tensorInput)
			# end
		# end

		self.modulePreprocess = Preprocess()

		self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFlow = []

		tensorFirst = [ self.modulePreprocess(tensorFirst) ]
		tensorSecond = [ self.modulePreprocess(tensorSecond) ]

		for intLevel in range(5):
			if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
				tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
				tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))
			# end
		# end

		tensorFlow = tensorFirst[0].new_zeros([ tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)) ])

		for intLevel in range(len(tensorFirst)):
			tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], Backward(tensorInput=tensorSecond[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled ], 1)) + tensorUpsampled
		# end

		return tensorFlow
	# end
# end

moduleNetwork = Network().cuda().eval()

##########################################################

def estimate(tensorFirst, tensorSecond):
	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

	#assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tensorFlow = torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tensorFlow[0, :, :, :].cpu()
# end

##########################################################

def getFileList(root):
	flow_root = root
        image_root = root[:-3]+'images/'
        print("Root directory: ", flow_root)
        file_list = sorted(glob(join(flow_root, '*/*.flo')))
        print(file_list)
        flow_list = []
        image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-6]
            fnum = int(fbase[-6:-4])
            print("File base: ",fbase)
            print("Image root: ",image_root)   
             
            img1 = join(flow_root, fprefix + "%02d"%(fnum+0) + '.png')
            img2 = join(flow_root, fprefix + "%02d"%(fnum+1) + '.png')
            
            print("Img1: ",img1)
            print("Img2: ",img2) 
            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            image_list += [[img1, img2]]
            flow_list += [file]
	return image_list, flow_list


#########################################################

if __name__ == '__main__':
	#tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
	#tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
	flow_dir = '/home/SDD_Work/ColorVersion'
        out_dir = '/home/SDD_Work/Results/pytorch_spynet'
        out_dir_summary = '/home/SDD_Work/Results/summary_pytorch_spynet'
	img_list, flow_list = getFileList(flow_dir)
        for i in range(0,len(img_list)):
		print("--")
		print(img_list[i][0])
		print(img_list[i][1])
		print(flow_list[i])
		tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(img_list[i][0]))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(img_list[i][1]))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		tensorOutput = estimate(tensorFirst, tensorSecond)           
                ## Create summary figure and save it
		flow = Flow()
		fig_file_name = join(out_dir_summary,flow_list[i][len(flow_dir)+1:-4]+'_Spynet_SintenClean_predicted.pdf')        
                if not os.path.exists(os.path.dirname(fig_file_name)):
                    os.mkdir(os.path.dirname(fig_file_name))
                gt_flow = flow._readFlow(flow_list[i])
                gt_flow_vis = flow._flowToColor(gt_flow)
                pred_flow = numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32)
                pred_flow_vis = flow._flowToColor(numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32))
                fig=plt.figure()
                img1 = numpy.array(PIL.Image.open(img_list[i][0]))
                img2 = numpy.array(PIL.Image.open(img_list[i][0]))
                head, img1_fname = os.path.split(img_list[i][0])
                head, img2_fname = os.path.split(img_list[i][1])
                epe_flow  = evaluate_flow(gt_flow,pred_flow)
                plt.subplot(221),plt.imshow(img1),plt.title('first frame')
                plt.subplot(222),plt.imshow(img2),plt.title('second frame')
                plt.subplot(223),plt.imshow(gt_flow_vis),plt.title('Ground Truth')
                plt.subplot(224),plt.imshow(pred_flow_vis),plt.title('Predicted OF: EPE'+str(epe_flow))
                plt.savefig(fig_file_name)
		plt.close()
		outfile_name = join(out_dir,flow_list[i][len(flow_dir)+1:-4]+'_Spynet_SintenClean_predicted.flo')        
                if not os.path.exists(os.path.dirname(outfile_name)):
                    os.mkdir(os.path.dirname(outfile_name))

                print(outfile_name)	
                objectOutput = open(outfile_name,'wb')

		numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
		numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
		numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)
		objectOutput.close()
		print("___")
# end
