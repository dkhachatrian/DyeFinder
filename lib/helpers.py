# -*- coding: utf-8 -*-

"""

Helpers for DyeFinder.

"""



import sys
from PIL import Image
import numpy as np
from matplotlib import colors

from matplotlib import pyplot as plt

#### To ensure the working directory starts at where the script is located...
import os
abspath = os.path.abspath(__file__) #in lib directory
dname = os.path.dirname(os.path.dirname(abspath)) #twice to get path to main directory
dep = os.path.join(dname, 'dependencies')
os.chdir(dname)
####

from collections import namedtuple

coord_info = namedtuple('coord_info', ['coord', 'orientation','coherence','energy'])



out_dir = os.path.join(dname, 'outputs') #directory for output files

cache_dir = os.path.join(dname, 'cache')

EPSILON = 0.05 #for RGB tuple values, providing a range of color to count as interesting
MAX = 255 #8-bit

DYE_IM = 1
ANISO_IM = 2



def get_image(im_flag):
    """
    Prompts user for name of image, looking in the dependencies directory.
    """
    while True:
        try:
            if im_flag == ANISO_IM:
                file_name = input("Please state the name of the file corresponding to the Text Images to be input, or enter nothing to quit: \n")
            elif im_flag == DYE_IM:
                file_name = input('Please input the filename corresponding to the stained image.')
            if file_name == '':
                sys.exit()
            im = Image.open(os.path.join(dep, file_name))
            break
        except FileNotFoundError:
            print("File not found! Please check the spelling of the filename input, and ensure the filename extension is written as well.")
            continue
        except IOError: #file couldn't be read as an Image
            print("File could not be read as an image! Please ensure you are typing the filename of the original image..")
            continue
        
    
    return file_name, im
    
    
def get_data():
    """
    Prompts user for names of files corresponding to outputs of OrientationJ's parameters: orientation, coherence, and energy.
    Input files are searched for in the script's dependencies folder.
    Input files must haev been saved as a Text Image using ImageJ.
    
    Returns a NumPy array of the data stacked such that the final axis has the data in order [orientation, coherence, energy].
    """

    data_names = ['orientation', 'coherence', 'energy']
    fnames = []
    data_list = []
    
    while len(fnames) < len(data_names):
        try:
            file_name = input("Please state the name of the file corresponding to the " + str(data_names[len(data_list)]) + " for the image of interest (saved as a Text Image from ImageJ), or enter nothing to quit: \n")
            with open(os.path.join(g.dep, file_name), 'r') as inf:
                d_layer = np.loadtxt(inf, delimiter = '\t')
            fnames.append(file_name)
            data_list.append(d_layer)
        except FileNotFoundError:
            print('File not found! Please ensure the name was spelled correctly and is in the dependencies directory.')
        except ValueError:
            print('File structure not recognized! Please ensure the file was spelled correctly and was saved as a Text Image in ImageJ.')
    
    data_shape = data_list[0].shape
    data_index = np.ndindex(data_shape)
    tupled_data = np.ndarray(data_shape).tolist()
    
    oris, cohs, eners = data_list
    
    for i in data_index:
        c_info = h.coord_info(coord=tuple(reversed(i)), orientation=oris[i], coherence=cohs[i], energy=eners[i])
        tupled_data[i] = c_info
        
    return tupled_data

    
    
    
    
def get_color_of_interest():
    """
    Prompt user for RGB value of color of interest.
    Returns an HSV tuple with values in range [0,1].
    """
    
    print('Please open your image in ImageJ and point your cursor to a pixel whose color is representative of the stain of interest.')
    
    while True:
        try:
            tup = input("Please type the RGB tuple corresponding to the pixel of interest. There should be three numbers, separated by commas, with values falling within [0,255]:\n")
            tup.strip('() ')
            c_rgb = [int(x)/MAX for x in tup.split(',')]
            if len(c_rgb) != 3:
                print('Error! Not enough numbers entered. Please try again.')
                continue
            elif min(c_rgb) < 0 or max(c_rgb) > 1:
                print('Error! Numbers does not fall within the expected range of [0,255]. Please try again.')
        except ValueError:
            print('Error! Numbers not entered. Please try again.')
            continue
    
    return colors.rgb_to_hsv(c_rgb)





def collect_coords_of_interest(image, color):
    """
    Collect coordinates in the image that is close to color (HSV tuple).
    Coords will be in data_array order, *not* in image (x,y) order.
    Returns the coordinates as a list.
    """
    
#    im = image.convert('RGB')
    im = image.convert('HSV') #HSV seems to be better at finding colors without error. At least, the 'H' part
    im_data = np.array(im) / MAX
    
    index = np.ndindex(im_data.shape[:-1]) #loop over (x,y(,z)) but not data itself
    
    coords = []
    
    for i in index:
        match = True
        zipped = zip(im_data[i], color)
        for e in zipped:
            if abs(e[1]-e[0]) >= EPSILON:
                match = False
                break
        if match == True:
            coords.append(i)            
            #coords.append(tuple(reversed(i))) #flipped from array order to image order
        
    return coords


def plot_histogram_data(data, coords, bins,predicate):
    """
    Bin datapoints corresponding to coordinates from a list (coords) to the data, according to a predicate function (predicate).
    The predicate function takes in a coord_info namedtuple, and an int for the number of bins.
    """
    
    