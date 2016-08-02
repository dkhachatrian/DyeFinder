# -*- coding: utf-8 -*-

"""
Globals for DyeFinder.
"""
#### To ensure the working directory starts at where the script is located...
import os
abspath = os.path.abspath(__file__) #in lib directory
dname = os.path.dirname(os.path.dirname(abspath)) #twice to get path to main directory
dep = os.path.join(dname, 'dependencies')
os.chdir(dname)
####

from collections import namedtuple


coord_labels = ['coord', 'orientation', 'coherence', 'energy']
label2val = {x:i for i,x in enumerate(coord_labels)} #choose things by name

coord_info = namedtuple('coord_info', coord_labels)



out_dir = os.path.join(dname, 'outputs') #directory for output files

cache_dir = os.path.join(dname, 'cache')

# create directories if necessary

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)
    



color2hsv_thresh = { 'brown': [(9,23),(63,209),(0,196)] } #

#EPSILON = 0.05 #for RGB tuple values, providing a range of color to count as interesting
MAX = 255 #8-bit

DYE_IM = 1
ANISO_IM = 2