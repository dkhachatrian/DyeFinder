# -*- coding: utf-8 -*-

"""
Functions that collect data or file locations, through user input or
pre-existing relationships, that are needed for the analysis.

@author: David G. Khachatrian
"""

from lib import globe as g
import sys
import os
import filecmp
import shutil

from lib.helpers import HelperException
from PIL import Image
import numpy as np

import pickle
from collections import defaultdict

#######################
##### USER INPUT ######
#######################


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
    (Has default location for MacOSX.)
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
            
            


##########################################
##### BATCH/FILE DEPENDENCIES INPUT ######
##########################################



def parse_dependencies(main_root = g.dep, im_ftypes = g.IMAGE_FILETYPES, ignore_starts_list = g.IGNORE_FILE_STARTS_LIST, ignore_dir_list = g.IGNORE_DIR_LIST):
    """
    For batch running of images in 'dependencies', set up directories in the 'outputs' folder.
    Returns a dictionary of (relative paths from dependencies) -->
    dictionaries of (prefix) --> g.prefix_info objects.
    
    Ignores files located in directories labeled '__ignore__'
    """
    
    rel_path2prefix_dict = {}
    
    all_files_have_info = True
    
    for root, dirs, files in os.walk(main_root, topdown = True):
        rel_path = os.path.relpath(root, main_root) #relative path from main_root to directory containing dirs and files
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
        

                
        # remember the way to get to this prefix's information
        rel_path2prefix_dict[rel_path] = prefix2fnames
        
    
    if not all_files_have_info:
        info_string = "Not all pairs of images had associated ROI_info.txt's!\
        Please fill out the .txt stubs and re-run this program."
        raise HelperException(info_string)
    
    return rel_path2prefix_dict



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
                    raise HelperException("Variable name '{0}' from file {1} not recognized by script as an expected variable!".format(parts[0], info_path))
                
    return info_dd
                



def get_image(im_flag, fpath = None):
    """
    Prompts user for name of image, looking in the dependencies directory.
    Returns the filename, and a PIL.Image of the associated file.
    
    If im_flag is set to g.BATCH, opens the image found at fpath.
    """

    if im_flag == g.BATCH:
        im_path = os.path.join(g.dep, fpath)
#        im = Image.open(im_path)
        file_name = os.path.basename(fpath)
        return file_name, im_path
    
    while True:
        if im_flag == g.ANISO_IM:
            file_name = input("Please state the name of the file corresponding to the Text Images to be input, or enter nothing to quit: \n")
        elif im_flag == g.DYE_IM:
            file_name = input('Please input the filename (in the dependencies folder) corresponding to the stained image:\n')
        if file_name == '':
            sys.exit()

        im_path = os.path.join(g.dep, file_name)
#        im = Image.open(im_path)
        if os.path.isfile(im_path):
            ok = False
            for ftype in g.IMAGE_FILETYPES:
                if im_path.endswith(ftype):
                    ok = True
                    break
            if ok:
                break
        else:
            print("File not found! Please check the spelling of the filename input, and ensure the filename extension is written as well.")
            continue
#        except FileNotFoundError:
#
#        except IOError: #file couldn't be read as an Image
#            print("File could not be read as an image! Please ensure you are typing the filename of the original image..")
#            continue
#        
    
    return file_name, im_path
    
    


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
        
        if len(data_list) != EXPECTED_LENGTH:
            raise HelperException("Not enough in .txt files found in {0} when running h.get_aniso_data in batch mode!".format(relpath))
    
    
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
        
    # TODO: combine pixels together (binning) to improve signal-to-noise ratio?
            
        
    return tupled_data





###############################
##### CACHE INPUT/OUTPUT ######
###############################



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
    
    
    coordname2label2vals = defaultdict(dict)
    
    dict_path = os.path.join(cache_dir, g.aggregate_label, 'dict.p')
    #load cache if available
    if os.path.exists(dict_path):
        with open(dict_path, mode = 'rb') as inf:
            coordname2label2vals = pickle.load(inf)
            return coordname2label2vals
    
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
        with open(dict_path, mode = 'wb') as out:
            pickle.dump(coordname2label2vals, out, pickle.HIGHEST_PROTOCOL)
            
    return coordname2label2vals



def write_dict_of_dicts_as_file(dd, out_path):
    """
    Given a dict of dict of lists, produces csv's at out_path (a directory)
    with values in separate columns.
    Each CSV file corresponds to one dictionary, labeled in the filename.
    
    Overwrites file without regard to previous existence or using replicate data.
    (So that if the cache is deleted, CSV files get updated with any new data
    that may have been placed that have the same prefix as a previous fileset.)
    """
    # Adapted from http://stackoverflow.com/questions/23613426/write-dictionary-of-lists-to-a-csv-file
    
    
    import csv
    

    
    # I prefer tab-delimited .txt's, which could be read by Excel
    out_label = "TXTs"
    file_format = 'txt'
    f_delimiter = '\t'
    
    try:
        os.makedirs(os.path.join(out_path,out_label))
    except os.error:
        pass #already exists

    
    for dict_label in dd:
        fname = "{0} dict.{1}".format(dict_label, file_format)
        d = dd[dict_label]
        
        csv_path = os.path.join(out_path, out_label, fname)
        
        
        with open(csv_path, "w") as outf:
            writer = csv.writer(outf, delimiter = f_delimiter) #should naturally work when opened in Excel now
#            writer.writerow(dict_label)
#                outf.write('{0}\n'.format(dict_label))
            value_labels = sorted(d.keys())
            writer.writerow(value_labels)
            writer.writerows(zip(*[d[key] for key in value_labels]))
            
#    with open(out_path, 'w') as outf:
#        for dict_label in dd:
#            


