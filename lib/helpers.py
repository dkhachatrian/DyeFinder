# -*- coding: utf-8 -*-

"""

Helpers for DyeFinder.

For the most part, this contains the data processing and visualization
components of the program. (Visualizations are output to file.)

"""



#import sys
from PIL import Image
import numpy as np
#from matplotlib import colors
#from matplotlib import cm
#import shutil
#import filecmp

import matplotlib.pyplot as plt
plt.ioff() #ensure no windows are opened while running the script
#from collections import defaultdict

#import pickle
import os
#import statistics as s
#import numpy as np

from lib import globe as g

#from matplotlib.colors import LogNorm


# so can throw exception at bad locations
class HelperException(Exception):
    pass










def find_outliers(data, z = 4):
    """
    Input: a NumPy array, corresponding to the intensity of its respective image.
    Returns: a list of those coordinates marked as outliers.
    In this case, an 'outlier' is more than z standard deviations above or below the mean.
    (In general, these intensity distributions seem to be strongly right-tailed...)
    """
#    nums = data.flatten().astype(int) #astype(int) 
#    mean = statistics.mean(data.flatten())
#    std = statistics.stdev(data.flatten())    
#    #std = statistics.stdev(data.flatten(), xbar = mean)
    mean,std = data.mean(), data.std()
    
    
    low,high = mean - z*std, mean + z*std
    
#    valid_mask = np.ndarray(data.shape, dtype = bool)
#    valid_mask.fill(True) #most will not be outliers

    outliers = []    
    
    indexer = np.ndindex(data.shape)
    
    for i in indexer:
        if data[i] < low or data[i] > high:
#            valid_mask[i] = False #marked as outlier
            outliers.append(i)
#        else:
#            valid_mask[i] = True
    
#    return valid_mask, outliers
    return outliers





def remove_coords(data, coords):
    """
    Removes from the dictionary of coord-->namedtuple (data) the coordinates contained in outlier_coords. Done in-place.
    """
    
    for c in coords:
        data.pop(c, None) #returns None instead of KeyError if already removed


def remove_background(data):
    """
    Removes data corresponding to the background of the image (histology slice).
    A coordinate is considered part of the background if its associated coherence and energy is less than or equal to g.EPSILON.
    Operation performed in-place.
    Returns coordinates of what was considered background.
    """
    popped = []
    for k in data:
        if data[k].coherence <= g.EPSILON and data[k].energy <= g.EPSILON:
            popped.append(k)
    for p in popped:
        data.pop(k, None) #returns None if already not in data

    return popped




    
    



def collect_coords_of_interest(image, ignore_list = None, color = 'brown'):
    """
    Collect coordinates in the image that is close to color (HSV tuple).
    Coords will be in data_array order, *not* in image (x,y) order.
    Coords will not contain any elements of ignore_list
    
    HSV thresholding bounds are obtained by manually performing the thresholding
    once (in e.g. ImageJ).
    These bounds are manually recorded in globe.color2hsv_thresh
    
    Returns the coordinates as a list.
    """
    
#    im = image.convert('RGB')
    im = image.convert('HSV') #HSV seems to be better at finding colors without error. At least, the 'H' part
    im_data = np.array(im) # [0,255]    
    #im_data = np.array(im) / MAX
    
    
    index = np.ndindex(im_data.shape[:-1]) #loop over (x,y(,z)) but not data itself

    im_data_dict = {i:im_data[i] for i in index}
    
    # remove unwanted coordinates
    for k in ignore_list:
        try:
            im_data_dict.pop(k)
        except KeyError:
            pass
    
    hsv_thresh = g.color2hsv_thresh[color]
    coords = []
    
    for i in im_data_dict:
#        if i in ignore_list:  #skip
#            continue
        match = True
        for band, thresh in zip(im_data[i], hsv_thresh):
            if band < min(thresh) or band > max(thresh):
                match = False
                break
        if match == True:
            coords.append(i)
        
    return coords




def get_measures(data, coords, measures):
    """
    Uses a list of functions (measures) to generate a dict of
    label-->list_of_values
    Returns this dict.
    """
    
    label2vals = {}
    
    for measure in measures:
        vals, label = measure(data,coords)
        label2vals[label] = vals
        
    return label2vals
    
    
def remove_low_values(data, epsilon):
    """
    If the energy or coherence of a coord in the dict data is below epsilon,
    remove it.
    More forceful than remove_background function, but allows for
    attempts to log-scale the data when doing plt.colorbar() for plt.hist2d().
    """
    pass
    

    
def plot_histogram_data(vals_dict, outdir, info, title_postfix, bins = 100, drange = None, aggregate_flag = False, log_flag = True):
    """
    Bin datapoints corresponding to coordinates from a list (coords) to the data, according to a predicate function (predicate).
    The predicate function takes in a coord_info namedtuple, and outputs a value to be used to build the histogram.
    plot_histogram_data returns the output of plt.hist()
    Also saves figure to outdir with name fname.
    """
    # maybe look into http://stackoverflow.com/questions/27156381/python-creating-a-2d-histogram-from-a-numpy-matrix
    
 
    #from matplotlib.colors import LogNorm
    
    # have clean figure/axes to plot new histogram on
    plt.cla()
    plt.clf()
    

    
    title_prefix = 'Histogram'
#    title_info = ' and '.join(labels)
#    title = ' '.join((title_prefix, title_info, title_postfix))
    
    title = ' '.join([title_prefix, title_postfix])
    
    #labels = list(reversed(labels)) #difference between array-like and image-like #not true for 1D or 2D hist!
#    
#    if len(predicates) == 2: #2d histogram
#        H, xedges, yedges = np.histogram2d(vals[0], vals[1], bins = bins, normed = True)
#        im = plt.imshow(H, interpolation='nearest', origin='low', \
#                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#    
#    
#    #gives a bin density when normed == True    
#    H, edges = np.histogramdd(vals_ll, bins = bins, normed = True)
#
#    fig = plt.figure()
#    fig.suptitle(title)
#    plt.xlabel(labels[0])
#    if len(predicates) == 2:
#        plt.ylabel(labels[1])
#    
#
    labels = sorted(list(vals_dict.keys())) #a way of keeping axes consistent across function calls
    

    
    
    #H, edges = np.histogramdd(vals_ll, bins = bins, normed = True)
    
    if len(labels) == 1: #1d histogram
        #redoing the binning, but it's worked in the past...
        low_bin_n = 0
        fig, ax = plt.subplots()
        vals = list(vals_dict.values())[0]
        ax.hist(vals, bins = bins, normed = True)
        ax.set_xlabel(labels[0])
        ax.set_title(title)
        
#        #already binned, so just put onto pyplot and save
#        # adapted from http://stackoverflow.com/questions/12303501/python-plot-simple-histogram-given-binned-data
#        dx = (max(drange)-min(drange)/edges[0].size)
#        plt.bar(left = np.arange(*drange, dx), height = H)
#        plt.xlabel(labels[0])
    
#    elif len(edges) == 2: #2d histogram
#        # adapted from http://matplotlib.org/examples/pylab_examples/colorbar_tick_labelling_demo.html   
#        edges = list(reversed(edges)) #flip to have it be (x,y)
#        #fig, ax = plt.subplots()
#        cax = ax.imshow(H, interpolation='nearest', cmap=cm.coolwarm)
#        #cax = ax.imshow(H, interpolation='nearest', cmap=cm.coolwarm, extent = [edges[0][0],edges[0][-1],edges[1][0],edges[1][1]])
#
#        # Add colorbar, make sure to specify tick locations to match desired ticklabels
#        cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
#        cbar.ax.set_yticklabels(['0', '0.5', '1'])  # vertically oriented colorbar
#        
#        # labels
#        
#        ax.set_xlabel(labels[0])
#        ax.set_ylabel(labels[1])
    
    
    elif len(labels) == 2:
        # adapted from http://matplotlib.org/examples/pylab_examples/hist2d_log_demo.html
        X = np.array(vals_dict[labels[0]])
        Y = np.array(vals_dict[labels[1]])
        # don't count bins with less than epsilon of the total number
        # (or that have nothing in them)
        epsilon = 0.0005
        low_bin_n = int(max(1, epsilon * len(X)))
        # ^too harsh?
        
#        low_bin_n = 1

#        if log_flag:
#            # log-transform the data
#            labels = ['log({0})'.format(l) for l in labels]
#            bins = 100
#            X_old, Y_old, X,Y = X, Y, np.log(X), np.log(Y) #log is base-2
#            # -log preserves "order" of val_small --> to the left/bottom of plot
#            low_bound = 1 #need at least one thing fall into the bin to show up

        
        if drange is None:
            # range will be between the 10th and 90th percentile
            lowp = .1
            highp = .9
            drange = [ [0,1], [0,1] ]
            for i, arr in enumerate((X,Y)):
                arr_sort = sorted(arr)
                low, high = int(lowp*len(arr_sort)), int(highp*len(arr_sort))
                drange[i][0], drange[i][1] = arr_sort[low], arr_sort[high]
#            x_sort = sorted(X)
#            minx,maxx = x_sort[lowp*len(x_sort)], x_sort[highp*len(x_sort)]
#            
#            y_sort = sorted(Y)
#            miny,maxy
#            
            
#            # log binning
#            if log_flag:
#                minx, miny = -6, -6
#                # -log_2(value) <= 6 with the resolution OrientationJ provides
#            else:
#                minx, miny = 0, 0
#            
#            if aggregate_flag:
#                maxx,maxy = max(X), max(Y)
#            else:
#                if log_flag:
#                    maxx,maxy = 0,0
#                else:
#                    maxx,maxy = 1,1
#            
##            minx,miny = 0,0
##            if log_flag:
##                maxx,maxy = 6,6 #basically nothing went past 6
##                # meaning, -log_2(value) <= 6 with the resolution OrientationJ provides
##            else:
##                maxx,maxy = 1,1 #falls between [0,1] naturally
##            
#            
#
##            if aggregate_flag:
##                drange = [[0, max(X)], [0, max(Y)]]
##            else:
##                if log_flag:
##                    drange = [[-10,0], [-10, 0]] #range captures values of [0.001,1]
##                else:
##                    drange = [[0,1],[0,1]]
#
#            drange = [[minx,maxx],[miny,maxy]]
            
            # no log-scaling
#            drange = [ [0,1] , [0,1] ]
        
#        if log_flag:
#            plt.hist2d(X,Y, bins = [bins,bins], range = drange, cmin = low_bound, norm = LogNorm())
#        else:
        plt.hist2d(X,Y, bins = [bins,bins], range = drange, cmin = low_bin_n)
        # plt.hist2d(vals_ll[0], vals_ll[1], bins = [bins,bins], normed = LogNorm())
        # plt.hist2d(vals_ll[0], vals_ll[1], bins=bins, normed=True)
        plt.colorbar()
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(title)

    fname = "{0} color_of_interest={1} {2} (n={3}) vals={6} {7}D histogram (bins={5}, low_bin_nin={8}).{4}".format(*info, bins, labels, len(labels), low_bin_n)     
  
    plt.savefig(os.path.join(outdir,fname))
    plt.close('all')
    #plt.close(fig) # remove figure from memory
        # because apparently garbage collection isn't a thing with plt/mpl...
    #return (H,edges)






def map_marked_pixels(outpath, coords, image_shape, fname):
    """
    Maps pixels marked by collect_coords_of_interest as black onto an otherwise white background.
    """
    #im = Image.new('L', image_shape)
    im_data = np.ndarray(tuple(reversed(image_shape))) #ensure shape matches im...
    im_data.fill(g.MAX) #make them all white...
    #except at coords
    for c in coords:
        im_data[c] = 0

    im = Image.fromarray(im_data.astype('uint8'))
    im.convert('1')
    
    fpath = os.path.join(outpath, fname)
    #fpath = '{0}/{1}'.format(outpath, 'dye_marked_pixels.png')
    im.save(fpath)
    #im.show() #debugging




                
def make_coords_list(d):
    """
    From a dictionary describing a rectangular ROI, return the corresponding 
    list of coordinates (as used in an array) (*not* pixel locations).
    """
    
    #corner = tuple(reversed(d[g.roi_var_names[0]]))
    corner = d[g.roi_var_names[0]]
    dx = d[g.roi_var_names[1]]
    dy = d[g.roi_var_names[2]]
    
    coords_list = []
    
    for i in range(dx):
        for j in range(dy):
            coord = (j + corner[1], i + corner[0]) #pixel order and coord order reversed
            coords_list.append(coord)
            
    return coords_list


## Note: get_coords is currently deprecated by manual delineation of the ROIs,
## fed into the program using data_input.get_ROI_info_from_txt
##
#def get_coords(data, data_mask, predicate, quant_flag = g.LOW):
#    """
#    Get coords in data that pass a predicate function.
#    data_mask is a data.shape NumPy array that states whether the coordinate in data is valid.
#    predicate takes three arguments: a g.aniso_tuple, a list of means, and a list of standard deviations
#    If too few (<0.001n) coordinates are chosen, instead use a quantile method.
#    """
#    # TODO: is data_mask deprecated? If so, remove from signature
##    if data.shape[:-1] != data_mask.shape:
##        raise HelperException("Data and data_mask's shape do not match as expected!")
##    indexer = np.ndindex(data_mask.shape)
##    
##    
##    #find mean and std of OK values
##    
##    
##    for i in indexer:
##        if data_mask[i]: #if not removed
#    
#    #find mean and std of remaining values
#    vals = ([c.orientation for c in data.values()], [c.coherence for c in data.values()], [c.energy for c in data.values()])
#    labels = ('orientation', 'coherence', 'energy')
#    means = {l:s.mean(v) for l,v in zip(labels,vals)}
#    stds = {l:s.stdev(v) for l,v in zip(labels, vals)}
#    #means = {'orientation': s.mean(oris), 'coherence': s.mean(cohs), 'energy':s.mean(eners)}
#    #means = g.coord_info(orientation=s.mean(oris), coherence = s.mean(cohs), energy = s.mean(eners))
#    #stds = {}
#    #stds = g.coord_info(orientation=s.stdev(oris, xbar = means.orientation), coherence = s.stdev(cohs, xbar = means.coherence), energy = s.stdev(eners, xbar = means.energy))
#    
#    coords = []
#    
#    for k in data:
#        if predicate(data[k], means, stds):
#            coords.append(k)
#
#    method = 'z_score'
#
#    DISC = 0.001    
#    
#    if len(coords) < DISC * len(data):
#        method = 'quantile'
#        coords = []
#        if quant_flag == g.LOW:
#            quants = (0.15, 0.25) # deciles
#        elif quant_flag == g.HIGH:
#            quants = (0.75, 0.85)
#        
#        #faster, by multiplying values of interest
#        anisos_sorted = sorted(data.values(), key = predicate)        
#        indices = (np.multiply(quants, len(anisos_sorted))).astype(int)
#        
#        coords = [tuple(reversed(x.coord)) for x in anisos_sorted[indices.min():indices.max()]] #flip back from x.coord (pixel location) to coord (as it would appear as the corresponding element in a NumPy array)
#        
#        # slow but most thorough
##        cohs_sorted = sorted(data.values(), key = lambda x: x.coherence)
##        eners_sorted = sorted(data.values(), key = lambda x: x.energy)
##        
##        indices = (np.multiply(quants, len(cohs_sorted))).astype(int)
##        
##        coords = [x.coord for x in cohs_sorted[indices.min():indices.max()] if x in eners_sorted] # O(n**2) ...
#        
#        
#        
#    
#    return coords, method
#    




#def has_low_aniso(aniso_tuple, means, stds, z_bounds = (1.5,2.5)):
#    """
#    Returns whether the tuple has low anisotropy, characterized by being within z_bounds standard deviations *below* the mean. Also passed in: means and standard deviations for orientation, coherence, energy.
#    """
#    z_proper = np.multiply(z_bounds, -1)
#    return has_high_aniso(aniso_tuple, means, stds, z_bounds = z_proper)
#
#def has_high_aniso(aniso_tuple, means, stds, z_bounds = (1.5,2.5)):
#    """
#    Returns whether the tuple has low anisotropy, characterized by high energy and high coherency.
#    Also passed in: means and standard deviations for orientation, coherence, energy.
#    """
#    adict = aniso_tuple._asdict()
#    #adict,mdict,sdict = aniso_tuple._asdict(), means._asdict(), stds._asdict() #maybe I can get around having to use a _'d method...
#    for k in adict:
#        if k == 'coord' or k == 'orientation':
#            continue #don't care about these for now
#        if not falls_in_zrange(adict[k], means[k], stds[k], z_bounds):
#        #if not falls_in_zrange(adict[k], mdict[k], sdict[k], z_bounds):
#            return False
#    #if got here, falls in proper range for all cases
#    return True
#    
#    
#
#def give_quantile_range(data, low, high):
#    """
#    Returns values between the low'th percentile and high'th percentile
#    """    
#    
#    
#def falls_in_zrange(val, mean, std, z_bounds):
#    return (val > (mean + std*min(z_bounds)) and val < (mean + std*max(z_bounds)))
#
#
#def dict2np_arr(d):
#    """
#    Converts a dictionary of (NumPy coordinates) --> coord_info namedtuple into a NumPy array ...
#    """
#    pass #not yet convinced I will actually need to perform this conversion...
