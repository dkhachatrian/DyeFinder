# -*- coding: utf-8 -*-

"""

Helpers for DyeFinder.

"""



import sys
from PIL import Image
import numpy as np
from matplotlib import colors
from matplotlib import cm
import shutil
import filecmp

import matplotlib.pyplot as plt
plt.ioff() #ensure no windows are opened while running the script
from collections import defaultdict

import pickle
import os
import statistics as s

from lib import globe as g


# so can throw exception at bad locations
class HelperException(Exception):
    pass


def prompt_user_to_set_up_files():
    """
    Gives user time to set up files before the script continues.
    """
    while True:
        print("Hello! Please set up the dependencies folder with the folder structure desired in the output directory.")
        print("Paired images (i.e., an NSC-stained image and its adjacent slice) should be saved with the same prefix and be saved with either '{0}.tif' or '{1}.tif' at the end, depending on whether it is the NSC-stained or membrane-stained slices, respectively.".format(g.NSC_LABEL, g.ANISO_LABEL))
        print("The paired images should have the same filename prefix, separated from the rest of the filename with a '{0}'".format(g.PREFIX_SEPARATOR))
        print("Currently, the image filenames cannot have parenthesis in them. (This breaks all ImageJ macros, as of version 1.51e.)")
        print("Type 'q' to quit. Otherwise, press Enter when the files are set up as desired:")
        uin = input()
        if uin == 'q':
            sys.exit()
        else:
            return


def get_ImageJ_location(platform):
    """
    Get absolute path to an instance of Fiji, with OrientationJ installed.
    Does not perform rigorous testing of whether you in fact pointed to a valid copy of ImageJ -- just checks that the user pointed to a file.
    """
    print("Hello! Please ensure your copy of ImageJ has the OrientationJ plugin installed.")
    if platform == 'win32': #windows
        while True:
            uinput = input("Please input the absolute path to your ImageJ executable, with directories separated by either only slashes ('/') or only backslashes ('\\'):\n")
            #first split
            parts = uinput.split('/')
            if len(parts) == 1:
                parts = uinput.split('\\')
                    

#            ijpath = os.sep.join(parts)
#            #because for some reason, os.path.join is silly about drive letters            
            
            
            #'manually' add in os.sep after drive letter (containing a ':')
            # otherwise, drive letter is not followed by os.sep
            # when using os.sep.join
            parts = [''.join([p, os.sep]) if ':' in p else p for p in parts]
            ijpath = os.path.join(*parts)
            #now re-split and os.path.join()
            #otherwise, drive letter is not followed by os.sep
            if os.path.isfile(ijpath):
                return ijpath
            else:
                print("File not found! Please ensure you separated the path directories with slashes ('/').")
    
    if platform == 'darwin': #mac osx
        relpath = os.path.join('Contents','MacOS', 'ImageJ-macosx') #from Fiji.app to actual executable
        while True:
            print("Please input the directory of the Fiji package, separated by slashes ('/').")
            print("Type 'q' and press Enter to quit.")
            print("Press Enter without typing anything to check the default location of '/Applications/Fiji.app':")
            uinput = input("Type your response now:\n")
            
            if uinput == '':
                uinput = '/Applications/Fiji.app'
            
            parts = uinput.split('/')
            
            
            
            # manually feed back in the first os.sep,
            # as the '' in parts is disregarded by os.path.join
            ijpath = os.sep + os.path.join(*parts, relpath)
            
            if os.path.isfile(ijpath):
                return ijpath
            else:
                print("File not found! Please ensure you separated the path directories with slashes ('/').")
            



def make_validity_mask(data, z = 4):
    """
    Input: a NumPy array, corresponding to the intensity of its respective image.
    Returns: a NumPy of type bool, with shape data.shape, where elements marked 'False' were deemed outliers and are to be ignored when performing further analysis. Also returns a list of those coordinates marked as outliers.
    In this case, an 'outlier' is more than z standard deviations above or below the mean.
    (In general, these intensity distributions seem to be strongly right-tailed...)
    """
#    nums = data.flatten().astype(int) #astype(int) 
#    mean = statistics.mean(data.flatten())
#    std = statistics.stdev(data.flatten())    
#    #std = statistics.stdev(data.flatten(), xbar = mean)
    mean,std = data.mean(), data.std()
    
    
    low,high = mean - z*std, mean + z*std
    
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

#def get_outliers(data):
#    """
#    Input: a NumPy array, corresponding to the intensity of its respective image.
#    Returns: a list of those coordinates marked as outliers.
#    In this case, an 'outlier' is above the 99.5th percentile.
#    """
#    indexer = np.ndindex(data)
#    num_tups = [] #will be L value, then original coordinate
#    
#    for i in indexer:
#        num_tups.append((data[i], i))
#    
#    num_tups = sorted(num_tups, key = lambda x: x[0]) #sort by luminance
#    
#    decile = 0.995
#    cutoff = int(decile*len(num_tups))
#    
#    outliers = [c[1] for c in num_tups[cutoff:]]
#    
#    return outliers
#    
    



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


def get_image(im_flag, fpath = None):
    """
    Prompts user for name of image, looking in the dependencies directory.
    Returns the filename, and a PIL.Image of the associated file.
    
    If im_flag is set to g.BATCH, opens the image found at fpath.
    """

    if im_flag == g.BATCH:
        im = Image.open(os.path.join(g.dep, fpath))
        file_name = os.path.basename(fpath)
        return file_name, im
    
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
    
    
def get_aniso_data(flag = None, root = None, relpath = None):
    """
    Prompts user for names of files corresponding to outputs of OrientationJ's parameters: orientation, coherence, and energy.
    Input files are searched for in the relative path from root.
    Input files must have been saved as a Text Image using ImageJ.
    
    Returns a dict of array coordinate-->coord_info (a namedtuple)
    
    If flag is set to g.BATCH, will look in relpath (relative to ./dependencies/) to find appropriate .txt files to construct the dictionary.
    """
    data_names = ['orientation', 'coherence', 'energy']
    EXPECTED_LENGTH = len(data_names)
    #fnames = []
    
    if flag == g.BATCH:
        data_list = {}
        fdir = os.path.join(root, relpath)
        #changing = True #will keep track of whether len(data_list) changes
        
        for fname in [x for x in os.listdir(fdir) if x.endswith('.txt')]:
            for label in data_names:
                if label in fname:
                    with open(os.path.join(fdir, fname), 'r') as inf:
                        d_layer = np.loadtxt(inf, delimiter = '\t')
                    #fnames.append(fname)
                    data_list[label] = d_layer
                    data_names.remove(label)
                    break
        
#        while changing:
#            if len(data_list) == len(data_names):
#                break
#            changing = False
#            for fname in os.listdir(fdir):
#                if fname.endswith('.txt') and data_names[len(data_list)] in fname:
#                    changing = True
#                    with open(os.path.join(fdir, fname), 'r') as inf:
#                        d_layer = np.loadtxt(inf, delimiter = '\t')
#                    fnames.append(fname)
#                    data_list.append(d_layer)
#                    break
        if len(data_list) != EXPECTED_LENGTH:
            raise HelperException("Not enough in .txt files found in {0} when running h.get_aniso_data in batch mode!".format(relpath))
    
    #else    
    else:
        data_list = {}
        while len(data_list) < EXPECTED_LENGTH:
            try:
                file_name = input("Please state the name of the file corresponding to the " + str(data_names[len(data_list)]) + " for the image of interest (saved as a Text Image from ImageJ), or enter nothing to quit: \n")
                with open(os.path.join(g.dep, file_name), 'r') as inf:
                    d_layer = np.loadtxt(inf, delimiter = '\t')
                #fnames.append(file_name)
                data_list[data_names[len(data_list)]] = d_layer
            except FileNotFoundError:
                print('File not found! Please ensure the name was spelled correctly and is in the dependencies directory.')
            except ValueError:
                print('File structure not recognized! Please ensure the file was spelled correctly and was saved as a Text Image in ImageJ.')
    
    #TODO: clean up this implementation...
    
    data_shape = data_list['orientation'].shape
    data_index = np.ndindex(data_shape)
    tupled_data = {} #dictionary of data_array_coordinate s --> coord_infos    
    #tupled_data = np.ndarray(data_shape).tolist()
    
    oris, cohs, eners = data_list['orientation'], data_list['coherence'], data_list['energy']
    
    for i in data_index: 
        c_info = g.coord_info(coord=tuple(reversed(i)), orientation=oris[i], coherence=cohs[i], energy=eners[i])
        tupled_data[i] = c_info#np.array(c_info)
        
    # TODO: combine pixels together to improve signal-to-noise ratio?
            
        
    return tupled_data

    
    
    
#    
#def get_color_of_interest():
#    """
#    Prompt user for RGB value of color of interest.
#    Returns an HSV tuple with values in range [0,1].
#    """
#    
#    print('Please open your image in ImageJ and point your cursor to a pixel whose color is representative of the stain of interest.')
#    
#    while True:
#        try:
#            tup = input("Please type the RGB tuple corresponding to the pixel of interest. There should be three numbers, separated by commas, with values falling within [0,255]:\n")
#            tup.strip('() ')
#            c_rgb = [int(x)/g.MAX for x in tup.split(',')]
#            if len(c_rgb) != 3:
#                print('Error! Not enough numbers entered. Please try again.')
#                continue
#            elif min(c_rgb) < 0 or max(c_rgb) > 1:
#                print('Error! Numbers does not fall within the expected range of [0,255]. Please try again.')
#                continue
#            break
#        except ValueError:
#            print('Error! Numbers not entered. Please try again.')
#            continue
#    
#    return tuple(colors.rgb_to_hsv(c_rgb))





def collect_coords_of_interest(image, ignore_list = None, color = 'brown'):
    """
    Collect coordinates in the image that is close to color (HSV tuple).
    Coords will be in data_array order, *not* in image (x,y) order.
    Coords will not contain any elements of ignore_list
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


#def plot_histogram_data(data, coords, outdir, fname, title, predicate, bins = 100, drange=(0,1)):
#    """
#    Bin datapoints corresponding to coordinates from a list (coords) to the data, according to a predicate function (predicate).
#    The predicate function takes in a coord_info namedtuple, and outputs a value to be used to build the histogram.
#    plot_histogram_data returns the output of plt.hist()
#    Also saves figure to outdir with name fname.
#    """
#    #TODO: use numpy.histogram2d or numpy.histogramdd instead of plt.hist    
#    
#    #from lib import hist_plotter    
#    
#    # clear old information so no overlap between successive calls
#    plt.cla()
#    plt.clf()    
#    
#    pred_data = []
#    
#    for c in coords:
#        pred_data.append(predicate(data[c]))
#    
#    # plot histogram using np.histogram so it doesn't force a new window to pop up
#    hist_data = plt.hist(pred_data, bins, range=drange)
##    hist, bin_edges = np.histogram(pred_data, bins, range=drange)
##    plt.bar(bin_edges[:-1], hist, width = 1)
##    plt.xlim(*drange)    
##    #plt.xlim(min(bin_edges), max(bin_edges))
#
#    plt.title(title)
#    #plt.show()
#    
#    hist_path = os.path.join(outdir, fname)
#    # TODO: provide labels to graph...
##    plt.show()
#    plt.savefig(hist_path)
##    with open(hist_path, 'w') as inf:
##    
#
#    
#    return hist_data
#    #return (hist, bin_edges)
#    
    
    
#def create_plot(coords_dict, plot_type = 'histogram', cache_flag = True, cache_loc = None):
#    """
#    Uses plot_type to determine which plotting function to call.
#    If cache_flag is set to True, saves pickled objects to specified cache_loc
#    (which must be provided).
#    
#    """
#    if cache_flag == True and cache_loc is None:
#        raise HelperException("create_plot was called with cache_flag set to True and cache_loc unspecified!")
#    
#
#    
    

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
    
    
    

    
def plot_histogram_data(vals_dict, outdir, info, title_postfix, bins = 100):
    """
    Bin datapoints corresponding to coordinates from a list (coords) to the data, according to a predicate function (predicate).
    The predicate function takes in a coord_info namedtuple, and outputs a value to be used to build the histogram.
    plot_histogram_data returns the output of plt.hist()
    Also saves figure to outdir with name fname.
    """
    # maybe look into http://stackoverflow.com/questions/27156381/python-creating-a-2d-histogram-from-a-numpy-matrix
    
 
    #from matplotlib.colors import LogNorm
    
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
    
    fname = "{0} color_of_interest={1} {2} (n={3}) vals={6} {7}D histogram (bins={5}).{4}".format(*info, bins, labels, len(labels))
    
    
    #H, edges = np.histogramdd(vals_ll, bins = bins, normed = True)
    
    if len(labels) == 1: #1d histogram
        #redoing the binning, but it's worked in the past...
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
        
        plt.hist2d(vals_dict[labels[0]], vals_dict[labels[1]], bins = [bins,bins], range = [[0,1],[0,1]], normed = True)
        # plt.hist2d(vals_ll[0], vals_ll[1], bins = [bins,bins], normed = LogNorm())
        # plt.hist2d(vals_ll[0], vals_ll[1], bins=bins, normed=True)
        plt.colorbar()
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(title)
        
     
    plt.savefig(os.path.join(outdir,fname))
    plt.close('all')
    #plt.close(fig) # remove figure from memory
        # because apparently garbage collection isn't a thing with plt/mpl...
    #return (H,edges)


def save_to_cache(var, info):
    """
    Saves variable to cache directory as a pickled file, for later inspection.
    """
    fpath = os.path.join(g.cache_dir, info)
    
    with open(fpath, mode='wb') as outf:
        pickle.dump(var, outf, pickle.HIGHEST_PROTOCOL)
    

def parse_dependencies(main_root = g.dep, im_ftypes = g.IMAGE_FILETYPES, ignore_starts_list = g.IGNORE_FILE_STARTS_LIST, ignore_dir_list = g.IGNORE_DIR_LIST):
    """
    For batch running of images in 'dependencies', set up directories in the 'outputs' folder.
    Returns a dictionary of (relative paths from dependencies) -->
    dictionaries of (prefix) --> g.prefix_info objects.
    
    Ignores files located in directories labeled '__ignore__'
    """
    
    # TODO: pair up images in lists here to ensure no mixup in rest of script,
    # when there is more than one set of images in one directory    
    
#    rel_paths, im_names_ll, related_info_ll, prefixes_ll = [], [], [], []
    
    rel_path2prefix_dict = {}
    #prefix2fnames = {} #prefix--> (list of im_names, list of related_info)
    
    #im_names_lll = []
    
    #im_names_lll will hold all im_names_ll
    #im_names_ll will hold pairs of images with the same beginning filename
    
    all_files_have_info = True
    
    for root, dirs, files in os.walk(main_root, topdown = True):
        rel_path = os.path.relpath(root, main_root) #relative path from main_root to directory containing dirs and files
#        im_names = []
#        info_names = []
        prefix2fnames = {}
        
        #remove values in ignore_list from dirs
        for i in ignore_dir_list:
            try:
                dirs.remove(i)
            except ValueError: #not in list
                pass

        for i in ignore_starts_list:
            files = [f for f in files if not f.startswith(i)]
        
            
        if len(files) == 0:
            continue
        #dirs = list(filter(lambda x: x not in ignore_list, dirs))
        #removes directories with the same name as any string in ignore_list 
        
#        if '__ignore__' in dirs:
#            dirs.remove('__ignore__')
#        prefixes = []
#        for f in files:
#            prefixes.append(f.split(' ')[0]) #get all prefixes
#            
        
        new_dirs = [g.out_dir, g.cache_dir]
        prefixes = set([f.split(' ')[0] for f in files])
        
        
        
        for prefix in prefixes:
            im_names = []
            info_names = []
            related_files = [f for f in files if f.startswith(prefix)]

            #make dir for each prefix -- separates the graphs
            for d in new_dirs:
                try:
                    os.makedirs(os.path.join(d, rel_path, prefix))
                except os.error:
                    pass #already exists
            
            
            info_file_exists = False
            
            for f in related_files:
#                # remove from files list to shorten it for next search (worth it?)
#                files.remove(f)
            
                # get related files (which I currently hardcode as .txt's)
                # make sure they changed the skeleton
                if f.endswith('.txt') and not filecmp.cmp(os.path.join(root,f), g.sample_roi_skeleton_path):
                    info_names.append(f)
                    info_file_exists = True
                    continue
                
                # get images
                for im_ftype in im_ftypes:
                    if f.endswith(im_ftype):

                        
                        im_names.append(f) #remember filenames found
                        break #check next file    
                
            
            if not info_file_exists:
                
                all_files_have_info = False
                #place .txt stub to be filled out
                txtname = "{0} ROI_info.txt".format(prefix)
                new_txt_path = os.path.join(root, txtname)
                
                shutil.copy(g.sample_roi_skeleton_path, new_txt_path)
                
            
            
            prefix2fnames[prefix] = g.prefix_info(im_fnames = im_names, info_fnames = info_names)
            
            
                
        #also make aggregrate output/cache dir for each relative path
        for d in new_dirs:
            try:
                os.makedirs(os.path.join(d,rel_path, g.aggregate_label))
            except os.error:
                pass
        
        
#        if len(files) > 0:
#            #see if there are .tif's (images to be processed) in the root dirctory
#            prefixes_seen = []
#            for f in files:
#                fparts = f.split(' ')
#                f_prefix = fparts[0]
#                
#                if f_prefix in prefixes_seen:
#                    continue
#                for im_ftype in im_ftypes:
#                    if f.endswith(im_ftype):
#                        #make dir for each directory containing tifs
#                        #outputs
#                        try:
#                            os.makedirs(os.path.join(g.out_dir, rel_path))
#                        except os.error:
#                            pass #already exists
#                        
#                        #cached files (e.g., .txt's)
#                        try:
#                            os.makedirs(os.path.join(g.cache_dir, rel_path))
#                        except os.error:
#                            pass #already exists
#                        
#                        im_names.append(f) #remember filenames found
#                        break #check next file
#                
#                prefixes_seen.append(fparts[0])
                
        
        rel_path2prefix_dict[rel_path] = prefix2fnames
        
#        rel_paths.append(rel_path)
#        im_names_ll.append(im_names)
#        related_info_ll.append(info_names)
#        prefixes_ll.append(prefixes)
    
    
    if not all_files_have_info:
        info_string = "Not all pairs of images had associated ROI_info.txt's!\
        Please fill out the .txt stubs and re-run this program."
        raise HelperException(info_string)
    
    return rel_path2prefix_dict
        
#    return rel_paths, im_names_ll, related_info_ll, prefixes_ll

#def recursive_defaultdict(dd = defaultdict, levels = 0, dtype = None):
#    """
#    Specify a (defaultdict of)*levels of dtype.
#    if levels < 1, simply returns dtype.
#    """
#    
#    if levels < 1:
#        return dtype
#    if levels == 1:
#        return defaultdict(dtype)
#    # else
#    return defaultdict(defaultdict, (levels - 1, dtype))
    


def load_all_vals(cache_dir, cache_flag = False):
    """
    Loads all pickled dictionaries within cache_dir to make one large dictionary,
    using the list of pickled labels in that same directory.
    Will be used to create a master dictionary of 'type of coordinate' --> 'value'
    for all images in a directory.
    Returns said dict.
    
    If cache_flag is set to True, saves the extended dictionary to relative
    aggregate directory.
    """
    
    #coordname2label2vals = defaultdict(defaultdict(list))
    
#    coordname2label2vals = defaultdict(lambda: defaultdict(list)) #will be dict of dict of lists
#    coordname2label2vals = recursive_defaultdict(levels = 2, dtype = list)
    #first, see if there's anything in the aggregate folder
    
#    for root, dirs, files in os.walk(cache_dir):
#        if root != os.path.join(cache_dir, g.aggregate_label):
#            continue
#        
#        if len(files) > 0:
#            if 'dict.p' in files:
#                with open(os.path.join(root, 'dict.p'), mode = 'rb') as inf:
#                    coordname2label2vals = pickle.load(inf)
#                    return coordname2label2vals
    
    coordname2label2vals = defaultdict(dict)
    
    for root, dirs, files in os.walk(cache_dir):
        try:
            dirs.remove(g.aggregate_label)
        except ValueError:
            pass
        
        relevant_files = [f for f in files if f.endswith('.p')] #only want pickled files
        
#        #first load labels
#        for f in relevant_files:
#            if f.endswith('list.p'):
#                with open(os.path.join(root,f), mode='rb') as inf:
#                    labels = pickle.load(inf)
#                    break
                
        if len(relevant_files) == 0:
            continue
        
        # load dictionary
        for f in relevant_files:
            if f.endswith('dict.p'):
                with open(os.path.join(root,f), mode='rb') as inf:
                    coord_name2vals_dict = pickle.load(inf)
                    break
                
        #extend masterdict
        for coord_name in coord_name2vals_dict:
            vals_dict = coord_name2vals_dict[coord_name]
            for val_label in vals_dict:
                coordname2label2vals[coord_name].setdefault(val_label, []).extend(vals_dict[val_label])
                # in the dict (value of defaultdict(dict)), if the label already
                # exists, extend the list by the subdict's values
                # otherwise, first set a default value of an empty list
                # then extend the list.
            
    if cache_flag:
        with open(os.path.join(cache_dir, g.aggregate_label, 'dict.p'), mode = 'wb') as out:
            pickle.dump(coordname2label2vals, out, pickle.HIGHEST_PROTOCOL)
            
    return coordname2label2vals


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



def get_ROI_info_from_txt(info_path):
    """
    Input:
    info = a path to a .txt file with specific formatting:
            - ignore lines starting with '#'
            - each new set of information is separated by '//' (and is labeled w)
            - names found in labels (in function body), then an equals sign, followed by expected values
    
    Output:
    a dict of dicts containing this information
    """
    import string
    #from collections import defaultdict
    #whitespace = ' \t\n'
    ignore_char = '#'
    newset_flag = '//'
    assigner = '='
    
    info_dict = {}
    info_dd = {}
    label = None
    
    
    
    with open(info_path, mode = 'r', encoding = 'utf8') as inf:
        for line in inf:
            
            if set(g.roi_var_names) == info_dict.keys() and label is not None:
                info_dd[label] = info_dict
                info_dict = {}
                label = None
                
            if line.startswith(ignore_char) or line == '\n':
                continue
            
            if newset_flag in line:
                # ensure there's a valid label
                parts = line.split(newset_flag)
                parts = [x.strip(string.whitespace) for x in parts]
                while True:
                    try:
                        parts.remove('')
                    except ValueError:
                        break
                
                label = '_'.join(parts)
            
            elif assigner in line:
                parts = line.split(assigner)
                parts = [x.strip(string.whitespace) for x in parts]
                
                assert len(parts) == 2
                
                # add to dict
                if parts[0] in g.roi_var_names:
                    info_dict[parts[0]] = eval(parts[1], {}, {})
                    #the two empty dicts correspond to globals and locals
                else:
                    raise HelperException("Variable name '{0}' from file {1} not recognized by script as an expected variable!".format(s[0], info_path))
                
    return info_dd
                
                
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
