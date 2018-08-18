
import math
import sys
import operator
import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance
import scipy.signal
import skimage
import skimage.io
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.optimize import minimize

def get_saliency_ft(img_path):

	# Saliency map calculation based on:

	img = skimage.io.imread(img_path)

	img_rgb = img_as_float(img)

	img_lab = skimage.color.rgb2lab(img_rgb) 

	mean_val = np.mean(img_rgb,axis=(0,1))

	kernel_h = (1.0/16.0) * np.array([[1,4,6,4,1]])
	kernel_w = kernel_h.transpose()

	blurred_l = scipy.signal.convolve2d(img_lab[:,:,0],kernel_h,mode='same')
	blurred_a = scipy.signal.convolve2d(img_lab[:,:,1],kernel_h,mode='same')
	blurred_b = scipy.signal.convolve2d(img_lab[:,:,2],kernel_h,mode='same')

	blurred_l = scipy.signal.convolve2d(blurred_l,kernel_w,mode='same')
	blurred_a = scipy.signal.convolve2d(blurred_a,kernel_w,mode='same')
	blurred_b = scipy.signal.convolve2d(blurred_b,kernel_w,mode='same')

	im_blurred = np.dstack([blurred_l,blurred_a,blurred_b])

	sal = np.linalg.norm(mean_val - im_blurred,axis = 2)
	sal_max = np.max(sal)
	sal_min = np.min(sal)
	sal = 255 * ((sal - sal_min) / (sal_max - sal_min))
	return sal