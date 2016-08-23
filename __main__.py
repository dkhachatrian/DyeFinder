# -*- coding: utf-8 -*-

"""
Finds pixels with specified color and coregisters them with anisotropy data. Produces a histogram showing distribution of pixels of interest with anisotropy information at said pixel.

@author: David G. Khachatrian
"""

from lib import helpers as h
from lib import data_io as dio
from lib import globe as g
from lib import aniso_measures as a
import numpy as np
import os
import subprocess
#from collections import namedtuple
import time
from functools import reduce
#from collections import defaultdict
import pickle
from PIL import Image


from sys import platform as _platform

#from lib import hist_plotter #not ever directly called, but changes to pyplot are necessary


np.set_printoptions(precision = 3)

imagej_loc = dio.get_ImageJ_location(_platform)
ij_macro_loc = os.path.join(g.lib, 'aniso_macro.ijm')
macro_label = 'macro_performed_list.txt'

oj_labels = ['coherence.txt', 'orientation.txt', 'energy.txt']
#postfix of txt's produced by ImageJ macro. Joined with filename by ' '
# NOT LINKED WITH MACRO. If macro changes, the above needs to change.
# 

dio.prompt_user_to_set_up_files()

#aniso_measures = h.choose_measures()

#aniso_measures = [a.weighted_anisotropy] #1d
aniso_measures = [a.coherence, a.energy] #2d

# remove_outliers = h.should_remove_outliers()
remove_outliers = False



relpath2prefix2info = dio.parse_dependencies()

# should the script treat the aggregates differently from individual dicts?
aggregates_different_flag = False

color_of_interest = 'brown'

should_cache_aggregate = False
should_cache_individual = False


#run ImageJ macros to get OrientationJ measures (?)

#for rel_path, dep_im_fnames, related_info_fnames in zip(rel_paths, dep_ims_fname_ll, dep_roi_infos, prefixes_ll):
#    if rel_path == '.': #same as trunk
#        rel_path = ''



for rel_path in relpath2prefix2info:
    prefix_dict = relpath2prefix2info[rel_path]
    
    
    print() #two newline spaces between directories
    print("Now working on files in {0}...".format(os.path.join(g.dep,rel_path)))    
    
    
    for prefix in prefix_dict:
        fnames = prefix_dict[prefix]
        dep_im_fnames = fnames.im_fnames
        related_info_fnames = fnames.info_fnames
    
        if len(dep_im_fnames) == 0: # no images to process
            continue    
    
        out_dir = os.path.join(g.out_dir, rel_path, prefix)
        rel_prefix_cache_dir = os.path.join(g.cache_dir, rel_path, prefix)   
        cached_dict_loc = os.path.join(rel_prefix_cache_dir, 'coord_name2vals_dict dict.p')


        print('Working on paired images: {0}...'.format(dep_im_fnames))
        
        #time each pair
        start_total = time.clock()
        


        if os.path.isfile(cached_dict_loc):
            with open(cached_dict_loc, mode = 'rb') as inf:
                coord_name2vals_dict = pickle.load(inf)
        
        else:
        
            info_ddl = []
        
            for f in related_info_fnames:
                info_path = os.path.join(g.dep, rel_path, f)
                info_ddl.append(dio.get_ROI_info_from_txt(info_path))
                #white_matter, gray_matter, injection_site = h.get_roi_info(info_path)
            
            
                
            
            macro_path = os.path.join(g.cache_dir,rel_path,prefix, macro_label)
            
            #create macro list file if necessary
            try:
                with open(macro_path, mode = 'x') as inf:
                    pass
            except FileExistsError:
                pass
        
        #    start = time.clock()
            # check if they have the same beginning. split with ' '
        #    paired_image_names_ll = [] #list of two-element lists
        #    for name1 in dep_im_fnames:
        #        for name2 in dep_im_fnames:
        #            if name1 == name2:
        #                continue
        #            paired = sorted([name1,name2]) #ensures same order --> can prevent duplicates
        #            if name1.split(g.PREFIX_SEPARATOR)[0] == name2.split(g.PREFIX_SEPARATOR)[0] and paired not in paired_image_names_ll:
        #                paired_image_names_ll.append(paired)
                        
        #            
        #    end = time.clock()
        #    print("Pairing image names took {0} seconds for a directory with {1} images.".format(end-start, len(dep_im_fnames)))
                    
            
        

    
            # TODO: preprocess images? e.g. resize to be smaller resolution and
            # improve signal-to-noise ratio    
    
    
            
            for dep_im_fname in dep_im_fnames:
                #if dep_im_fname.endswith('.tif'):
                if g.ANISO_LABEL in dep_im_fname:
#                    aniso_fname, aniso_im = dio.get_image(g.BATCH, fpath = os.path.join(rel_path, dep_im_fname))
                    aniso_fname, aniso_im_path = dio.get_image(g.BATCH, fpath = os.path.join(rel_path, dep_im_fname))
                    
                    # generate the Text Images if not already created
#                    im_path = os.path.join(g.dep, rel_path, aniso_fname)
                    im_path = aniso_im_path
                    
                    #check to see if the file's been processed
                 #   if macro_label not in os.listdir(os.path.join(g.dep,rel_path)): 
                    # 'a+' allows for appending and reading, *creating file if non-existent*
                    with open(macro_path, mode = 'r+') as inf:
                        #inf.seek(0) #go to beginning of file
                        found = False
                        for line in inf:
                            stripped = line.strip('\n')
                            if stripped == im_path:
                                found = True
                                break
                    if not found:
                    #if macro_label not in os.listdir(os.path.join(g.dep,rel_path)):        
                        sub_args = [imagej_loc, '--headless', '-macro', ij_macro_loc, im_path]
        #                args = ['java', '-jar', 'ij.jar', '-batch', 'aniso macro', im_path]
        #                    # will require aniso_macro.ijm to be installed as a macro in fiji
                        subprocess.run(sub_args, check = True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
                        #creates the txt files
                        #suppressing output...
                        #will throw subprocess.CalledProcessError exception if the subprocess fails for some reason...
                        #but if not, hopefully that means the macro also worked correctly
                        #so let's mark that it ran here on a specific file
                        with open(macro_path, 'a+') as inf:
                            inf.write('{0}\n'.format(im_path))
                        
                        # let's move the output .txt's to the cache directory
                        for label in oj_labels:
                            txt_name = ' '.join([aniso_fname, label])
                            from_path = os.path.join(g.dep, rel_path, txt_name)
                            to_path =  os.path.join(g.cache_dir, rel_path, prefix, txt_name)
                            os.rename(from_path, to_path)
                        
                    
                elif g.NSC_LABEL in dep_im_fname:
                    dye_fname, dye_im_path = dio.get_image(g.BATCH, fpath = os.path.join(rel_path, dep_im_fname))
                    dye_im = Image.open(dye_im_path)
        
        
        
#            im_data_shape = tuple(reversed(dye_im.size))
            
            f_prefix = dye_fname
            
        
        
        
        
        
            
            
    #        Z = 4
            
            #anisotropy...
            
            
            
            start = time.clock()
            aniso_data = dio.get_aniso_data(flag = g.BATCH, root = g.cache_dir, relpath = os.path.join(rel_path, prefix)) #for anisotropy data
            #aniso_fname, aniso_im = h.get_image(g.ANISO_IM)
            end = time.clock()
            print("get_aniso_data took {0} seconds for an image with {1} pixels.".format(end-start, len(aniso_data)))
            
            
            h.remove_low_values(data = aniso_data, epsilon = 0.001) 
            
            
            
            outlier_coords = [] # currently not dealing with outliers
            # TODO: remove line-like image artifacts
            
            
            # for when using quantile method of getting outliers...
            #validity_mask = None  #filler
            #outlier_coords = h.get_outliers(np.array(aniso_im.convert('L')))
            if remove_outliers:
                h.remove_coords(data = aniso_data, coords = outlier_coords)
                
            bg_coords = h.remove_background(data = aniso_data)
            ignore_coords = bg_coords + outlier_coords
            
            
            #color_of_interest = h.get_color_of_interest()
    
            
            
            
            
            start = time.clock()
            coords_of_interest = h.collect_coords_of_interest(dye_im, ignore_list = ignore_coords, color = color_of_interest)
            #coords_of_interest = h.collect_coords_of_interest(dye_im, color = color_of_interest) #future feature?
            end = time.clock()
            print("coords_of_interest pruning took {0} seconds for an image with {1} coordinates in the ignore_list.".format(end-start, len(ignore_coords)))
            

            # not the most refined ingesting of info ...
            # currently operating on the 'one info.txt per pair' mindset
            # (to develop this quickly)
        
            # very lazy method of handling unexpected numbers of related txt's...
            roi_info_dd = info_ddl[0]
            
        #    assert len(info_ddl) == 0    
        #    
        #    for info_dd in info_ddl:
        #        if 'ROI_info' in info:
        #            roi_info_dd = h.get_ROI_coords(info)
        #            break
            
            for label in roi_info_dd:
                roi_info_dd[label]['coords'] = h.make_coords_list(roi_info_dd[label])
                
            
            
            
            
            # copy NSC locations, remove injection site NSCs
            # using roundabout way to optimize time (instead of comparing two lists)
            dye_coords_no_ij = {c:' ' for c in coords_of_interest}
            
            
            for c in roi_info_dd['INJECTION_SITE']['coords']:
                dye_coords_no_ij.pop(c, None) #O(1)
            
            dye_coords_no_ij = list(dye_coords_no_ij.keys())
            
            #naming histograms...
            coords_dict = {'dye_coords': coords_of_interest, \
            'dye_coords_no_ij': dye_coords_no_ij, \
            'all_coords_(no_bg_or_artifacts)': aniso_data.keys()}        
            
            for label in roi_info_dd:
                coords_dict[label] = roi_info_dd[label]['coords']
            
            
            coord_name2vals_dict = {}
            
            #build dict
            for coords_name in coords_dict:
                coords = coords_dict[coords_name]
                coord_name2vals_dict[coords_name] = h.get_measures(data = aniso_data, coords = coords, measures = aniso_measures)



            #unload the fairly large dict to allow for garbage collection
            aniso_data = None
            

            h.map_marked_pixels(outpath = out_dir, coords = coords_of_interest, image_shape = dye_im.size, fname = '{0} dye_marked_pixels.png'.format(f_prefix))
            h.map_marked_pixels(outpath = out_dir, coords = bg_coords, image_shape = dye_im.size, fname = '{0} bg_pixels.png'.format(f_prefix)) #debugging
            #h.map_marked_pixels(outpath = out_dir, coords = outlier_coords, image_shape = dye_im.size, fname = '{0} outlier_pixels z={1}.png'.format(f_prefix, Z)) #debugging

            
            #dump dict and labels, to refind appropriate values later
            # for large images, dicts may be obscenely large
            # NOTE: disabling caching of the individual dicts
            # will make aggregation through the script impossible!
            # However, will still export to CSV (see below)
            if should_cache_individual:
                with open(cached_dict_loc, mode = 'wb') as out:
                    pickle.dump(coord_name2vals_dict,out,pickle.HIGHEST_PROTOCOL)
            
            #end manual dict creation
            
            
        start = time.clock()       
        dio.write_dict_of_dicts_as_file(dd = coord_name2vals_dict, out_path = out_dir)   
        end = time.clock()
        print("Writing CSV's into {0} took {1} seconds.".format(out_dir, end-start))        
            
        #now plot the dict (either made from scratch or loaded from pickled file)
        for label in coord_name2vals_dict:
            vals = coord_name2vals_dict[label]
            n_bins = 100
            hist_info = ['individual_image', color_of_interest, label, len(vals['coherence']), g.hist_ftype]
            #hist_info = "dye_imagename={0} color_of_interest={1} {2} (n={3}) histogram (bins={4}).{5}".format(dye_fname, color_of_interest, coords_name, len(coords), n_bins, hist_ftype) #filename
            title_p = "for specified group: {0}".format(label)
            
            h.plot_histogram_data(vals_dict = coord_name2vals_dict[label], outdir = out_dir, info = hist_info, title_postfix = title_p, bins = n_bins)
            
        coord_name2vals_dict = {} #release data            
        
        end_total = time.clock()
        
        print("It took {0} seconds to perform the operations on the paired image list {1}.".format(end_total-start_total, dep_im_fnames))
        print() #one newline space between paired image lists


    out_dir = os.path.join(g.out_dir, rel_path, g.aggregate_label)
    # aggregate across entire directory
    rel_cache_dir = os.path.join(g.cache_dir, rel_path)
    
    #check if there are cached .p files in the relative cache directory
    # if there isn't, can't build aggregate dictionary    
    found_pickled_dict = False
    for root, dirs, files in os.walk(rel_cache_dir, topdown = True):
        if found_pickled_dict:
            break
        for f in files:
            if f.endswith('dict.p'): #found at least one
                found_pickled_dict = True
                break
    
    if found_pickled_dict: #perform aggregate dict activities
        start = time.clock()
        coord_name2vals_dict = dio.load_all_vals(rel_cache_dir, cache_flag = should_cache_aggregate)
        end = time.clock()
        print("Creating/loading the aggregate dictionary for {0} took {1} seconds.".format(rel_path, end-start))
        
        
        start = time.clock()       
        dio.write_dict_of_dicts_as_file(dd = coord_name2vals_dict, out_path = out_dir)   
        end = time.clock()
        print("Writing CSV's into {0} took {1} seconds.".format(out_dir, end-start))    
        
    
        for label in coord_name2vals_dict:
            vals = coord_name2vals_dict[label]
            n_bins = 100
            hist_info = ['aggregate', color_of_interest, label, len(vals['coherence']), g.hist_ftype]
            #hist_info = "dye_imagename={0} color_of_interest={1} {2} (n={3}) histogram (bins={4}).{5}".format(dye_fname, color_of_interest, coords_name, len(coords), n_bins, hist_ftype) #filename
            title_p = "for specified group: {0}".format(label)
            
            h.plot_histogram_data(vals_dict = coord_name2vals_dict[label], outdir = out_dir, info = hist_info, title_postfix = title_p, bins = n_bins, aggregate_flag = aggregates_different_flag) #, aggregate_flag = True


print('Done!')