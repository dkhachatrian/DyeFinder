# -*- coding: utf-8 -*-

"""

Helpers for DyeFinder.

"""



import sys
from PIL import Image
import numpy as np
from matplotlib import colors

from matplotlib import pyplot as plt

import pickle
import os
import statistics
import statistics as s

from lib import globe as g


# so can throw exception at bad locations
class HelperException(Exception):
    pass



def make_validity_mask(data, z = 2.8):
    """
    Input: a NumPy array, corresponding to the intensity of its respective image.
    Returns: a NumPy of type bool, with shape data.shape, where elements marked 'False' were deemed outliers and are to be ignored when performing further analysis. Also returns a list of those coordinates marked as outliers.
    In this case, an 'outlier' is more than z standard deviations above or below the mean.
    (In general, these intensity distributions seem to be strongly right-tailed...)
    """
    
    mean = statistics.mean(data)
    std = statistics.stdev(data, xbar = mean)
    
    
    low,high = mean-z*std, mean+z*std
    
    valid_mask = np.ndarray(data.shape, dtype = bool)
    valid_mask.fill(True) #most will not be outliers
    
    outliers = []    
    
    indexer = np.ndindex(valid_mask.shape)
    
    for i in indexer:
        if data[i] < low or data[i] > high:
            valid_mask[i] = False #marked as outlier
            outliers.append(i)
#        else:
#            valid_mask[i] = True
    
    return valid_mask, outliers



def remove_outliers(data, outlier_coords):
    """
    Removes from the dictionary of coord-->namedtuple (data) the coordinates contained in outlier_coords. Done in-place.
    """
    
    for c in outlier_coords:
        data.pop(c)


def remove_background(data):
    """
    Removes data corresponding to the background of the image (histology slice).
    A coordinate is considered part of the background if its associated coherence and energy is less than or equal to g.EPSILON.
    Operation performed in-place.
    """
    
    for k in data:
        if data[k].coherence <= g.EPSILON and data[k].energy <= g.EPSILON:
            data.pop(k)



def get_image(im_flag):
    """
    Prompts user for name of image, looking in the dependencies directory.
    """
    while True:
        try:
            if im_flag == g.ANISO_IM:
                file_name = input("Please state the name of the file corresponding to the Text Images to be input, or enter nothing to quit: \n")
            elif im_flag == g.DYE_IM:
                file_name = input('Please input the filename (in the dependencies folder) corresponding to the stained image:\n')
            if file_name == '':
                sys.exit()
            im = Image.open(os.path.join(g.dep, file_name))
            break
        except FileNotFoundError:
            print("File not found! Please check the spelling of the filename input, and ensure the filename extension is written as well.")
            continue
        except IOError: #file couldn't be read as an Image
            print("File could not be read as an image! Please ensure you are typing the filename of the original image..")
            continue
        
    
    return file_name, im
    
    
def get_aniso_data():
    """
    Prompts user for names of files corresponding to outputs of OrientationJ's parameters: orientation, coherence, and energy.
    Input files are searched for in the script's dependencies folder.
    Input files must haev been saved as a Text Image using ImageJ.
    
    Returns a dict of array coordinate-->coord_info (a namedtuple)
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
    
    #TODO: clean up this implementation...
    
    data_shape = data_list[0].shape
    data_index = np.ndindex(data_shape)
    tupled_data = {} #dictionary of data_array_coordinate s --> coord_infos    
    #tupled_data = np.ndarray(data_shape).tolist()
    
    oris, cohs, eners = data_list
    
    for i in data_index: 
        c_info = g.coord_info(coord=tuple(reversed(i)), orientation=oris[i], coherence=cohs[i], energy=eners[i])
        tupled_data[i] = np.array(c_info)
        
    # TODO: combine pixels together to improve signal-to-noise ratio?
            
        
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
            c_rgb = [int(x)/g.MAX for x in tup.split(',')]
            if len(c_rgb) != 3:
                print('Error! Not enough numbers entered. Please try again.')
                continue
            elif min(c_rgb) < 0 or max(c_rgb) > 1:
                print('Error! Numbers does not fall within the expected range of [0,255]. Please try again.')
                continue
            break
        except ValueError:
            print('Error! Numbers not entered. Please try again.')
            continue
    
    return tuple(colors.rgb_to_hsv(c_rgb))





def collect_coords_of_interest(image, color = 'brown'):
    """
    Collect coordinates in the image that is close to color (HSV tuple).
    Coords will be in data_array order, *not* in image (x,y) order.
    Returns the coordinates as a list.
    """
    
#    im = image.convert('RGB')
    im = image.convert('HSV') #HSV seems to be better at finding colors without error. At least, the 'H' part
    im_data = np.array(im) # [0,255]    
    #im_data = np.array(im) / MAX
    
    index = np.ndindex(im_data.shape[:-1]) #loop over (x,y(,z)) but not data itself
    hsv_thresh = g.color2hsv_thresh[color]
    coords = []
    
    for i in index:
        match = True
        for band, thresh in zip(im_data[i], hsv_thresh):
            if band < min(thresh) or band > max(thresh):
                match = False
                break
        if match == True:
            coords.append(i)
#        match = True
#        zipped = zip(im_data[i], color)
#        for e in zipped:
#            if abs(e[1]-e[0]) >= EPSILON: #comparing each element pairwise
#                match = False
#                break
#        if match == True:
#            coords.append(i)            
#            #coords.append(tuple(reversed(i))) #flipped from array order to image order
        
    return coords


def plot_histogram_data(data, coords, fname, title, predicate, bins = 100, drange=(0,1)):
    """
    Bin datapoints corresponding to coordinates from a list (coords) to the data, according to a predicate function (predicate).
    The predicate function takes in a coord_info namedtuple, and outputs a value to be used to build the histogram.
    plot_histogram_data returns the output of plt.hist()
    Also saves figure to 'outputs' directory.
    """
    #TODO: use numpy.histogram2d or numpy.histogramdd instead of plt.hist    
    
    pred_data = []
    
    for c in coords:
        pred_data.append(predicate(data[c]))
    
    # plot histogram using np.histogram so it doesn't force a new window to pop up
    hist_data = plt.hist(pred_data, bins, range=drange)
#    hist, bin_edges = np.histogram(pred_data, bins, range=drange)
#    plt.bar(bin_edges[:-1], hist, width = 1)
#    plt.xlim(*drange)    
#    #plt.xlim(min(bin_edges), max(bin_edges))

    plt.title(title)
    #plt.show()
    
    hist_path = os.path.join(g.out_dir, fname)
    # TODO: provide labels to graph...
#    plt.show()
    plt.savefig(hist_path)
#    with open(hist_path, 'w') as inf:
#    
    
    return hist_data
    #return (hist, bin_edges)
    
    
def weighted_anisotropy(aniso_tuple):
    """
    Returns a coherence-weighted energy value.
    """
    return aniso_tuple[g.label2val['coherence']] * aniso_tuple[g.label2val['energy']]



def save_to_cache(var, info):
    """
    Saves variable to cache directory as a pickled file, for later inspection.
    """
    fpath = os.path.join(g.cache_dir, info)
    
    with open(fpath, mode='wb') as outf:
        pickle.dump(var, outf, pickle.HIGHEST_PROTOCOL)
    

def set_up_outputs():
    """
    For batch running of images in 'dependencies', set up directories in the 'outputs' folder.
    Returns the path to the image, relative to '/dependencies/'
    """
    im_names = []
    for root, dirs, files in os.walk(g.dep, topdown = True):
        if len(files) > 0:
            #see if there are .tif's (images to be processed) in the root dirctory
            for f in files:
                if '.tif' in f:
                    #make dir for each directory containing tifs, in outputs
                    rel_path = os.path.relpath(root, g.dep)
                    try:
                        os.makedirs(os.path.join(g.out_dir, rel_path))
                    except os.error:
                        pass #already exists
                    
                    im_names.append(os.path.join(rel_path, f)) #remember filepath
        
    return im_names


def map_marked_pixels(outpath, coords, image_shape):
    """
    Maps pixels marked by collect_coords_of_interest onto an otherwise white background.
    """
    #im = Image.new('L', image_shape)
    im_data = np.ndarray(tuple(reversed(image_shape))) #ensure shape matches im...
    im_data.fill(g.MAX) #make them all white...
    #except at coords
    for c in coords:
        im_data[c] = 0

    im = Image.fromarray(im_data.astype('uint8'))
    im.convert('1')
    
    fpath = os.path.join(outpath, 'dye_marked_pixels.png')
    #fpath = '{0}/{1}'.format(outpath, 'dye_marked_pixels.png')
    im.save(fpath)
    #im.show() #debugging


def get_coords(data, data_mask, predicate, quant_flag = g.LOW):
    """
    Get coords in data that pass a predicate function.
    data_mask is a data.shape NumPy array that states whether the coordinate in data is valid.
    predicate takes three arguments: a g.aniso_tuple, a list of means, and a list of standard deviations
    If too few (<0.001n) coordinates are chosen, instead use a quantile method.
    """
    # TODO: is data_mask deprecated? If so, remove from signature
#    if data.shape[:-1] != data_mask.shape:
#        raise HelperException("Data and data_mask's shape do not match as expected!")
#    indexer = np.ndindex(data_mask.shape)
#    
#    
#    #find mean and std of OK values
#    
#    
#    for i in indexer:
#        if data_mask[i]: #if not removed
    
    #find mean and std of remaining values
    oris, cohs, eners = [c.orientation for c in data.values], [c.coherence for c in data.values], [c.energy for c in data.values]
    means = g.coord_info(orientation=s.mean(oris), coherence = s.mean(cohs), energy = s.mean(eners))
    stds = g.coord_info(orientation=s.stdev(oris, xbar = means.orientation), coherence = s.stdev(cohs, xbar = means.coherence), energy = s.stdev(eners, xbar = means.energy))
    
    coords = []
    
    for k in data:
        if predicate(data[k], means, stds):
            coords.append(k)

    method = 'z_score'

    DISC = 0.001    
    
    if len(coords) < DISC * len(data):
        method = 'quantile'
        coords = []
        if quant_flag == g.LOW:
            quants = (0.15, 0.25) # deciles
        elif quant_flag == g.HIGH:
            quants = (0.75, 0.85)
        
        #faster, by multiplying values of interest
        anisos_sorted = sorted(data.values(), key = weighted_anisotropy)        
        indices = (np.multiply(quants, len(anisos_sorted))).astype(int)
        
        coords = [x.coord for x in anisos_sorted[indices.min():indices.max()]]
        
        # slow but most thorough
#        cohs_sorted = sorted(data.values(), key = lambda x: x.coherence)
#        eners_sorted = sorted(data.values(), key = lambda x: x.energy)
#        
#        indices = (np.multiply(quants, len(cohs_sorted))).astype(int)
#        
#        coords = [x.coord for x in cohs_sorted[indices.min():indices.max()] if x in eners_sorted] # O(n**2) ...
        
        
        
    
    return coords, method
    




def has_low_aniso(aniso_tuple, means, stds, z_bounds = (1.5,2.5)):
    """
    Returns whether the tuple has low anisotropy, characterized by being within z_bounds standard deviations *below* the mean. Also passed in: means and standard deviations for orientation, coherence, energy.
    """
    z_proper = np.multiply(z_bounds, -1)
    return has_high_aniso(aniso_tuple, means, stds, z_bounds = z_proper)

def has_high_aniso(aniso_tuple, means, stds, z_bounds = (1.5,2.5)):
    """
    Returns whether the tuple has low anisotropy, characterized by high energy and high coherency.
    Also passed in: means and standard deviations for orientation, coherence, energy.
    """
    adict,mdict,sdict = aniso_tuple._asdict(), means._asdict(), stds._asdict() #maybe I can get around having to use a _'d method...
    for k in adict:
        if k == 'coord' or k == 'orientation':
            continue #don't care about these for now
        if not falls_in_zrange(adict[k], mdict[k], sdict[k], z_bounds):
            return False
    #if got here, falls in proper range for all cases
    return True
    
    

def give_quantile_range(data, low, high):
    """
    Returns values between the low'th percentile and high'th percentile
    """    
    
    
def falls_in_zrange(val, mean, std, z_bounds):
    return (val > (mean + std*min(z_bounds)) and val < (mean + std*max(z_bounds)))


def dict2np_arr(d):
    """
    Converts a dictionary of (NumPy coordinates) --> coord_info namedtuple into a NumPy array ...
    """
    pass #not yet convinced I will actually need to perform this conversion...
