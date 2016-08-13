# -*- coding: utf-8 -*-

"""
Globals for DyeFinder.
"""
#### To ensure the working directory starts at where the script is located...
import os
abspath = os.path.abspath(__file__) #in lib directory
dname = os.path.dirname(os.path.dirname(abspath)) #twice to get path to main directory
dep = os.path.join(dname, 'dependencies')
lib = os.path.join(dname, 'lib')
os.chdir(dname)
####

from collections import namedtuple


coord_labels = ['coord', 'orientation', 'coherence', 'energy']
label2val = {x:i for i,x in enumerate(coord_labels)} #choose things by name

coord_info = namedtuple('coord_info', coord_labels)

ANISO_LABEL = 'aniso' #in filename, for DiI stain
NSC_LABEL = 'nsc_locs' #in filename, for brown stain


IMAGE_FILETYPES = ['.tif']

#directories with labels in this list will be ignored
# by h.set_up_outputs()
IGNORE_LIST = ['__ignore__']


PREFIX_SEPARATOR = ' '


ROI_LABELS = ['WHITE_MATTER', 'GRAY_MATTER', 'INJECTION_SITE']
roi_var_names = ['top_left_corner_pixel', 'length', 'width']


out_dir = os.path.join(dname, 'outputs') #directory for output files

cache_dir = os.path.join(dname, 'cache')

ignore_dir = os.path.join(dep, '__ignore__')


# create directories if necessary

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)

if not os.path.isdir(dep):
    os.mkdir(dep)
    
if not os.path.isdir(ignore_dir):
    os.mkdir(ignore_dir)
    

EPSILON = 0.01 #for use in removing background


# used in h.get_coords_of_interest() to get pixels with a specific color
# HSV tuple bounds should be obtained manually once, then added here
# with order color_name-->[H_bounds,S_bounds,V_bounds]
color2hsv_thresh = { 'brown': [(9,23),(63,209),(0,196)] } 


#EPSILON = 0.05 #for RGB tuple values, providing a range of color to count as interesting
MAX = 255 #8-bit

DYE_IM = 1
ANISO_IM = 2
BATCH = 3


LOW = 'l'
HIGH = 'h'