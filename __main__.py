# -*- coding: utf-8 -*-

"""

Finds pixels with specified color and coregisters them with anisotropy data. Produces a histogram showing distribution of pixels of interest with anisotropy information at said pixel.

"""

from lib import helpers as h
from lib import globe as g
import numpy as np
import os
import subprocess
#from collections import namedtuple


from sys import platform as _platform

#from lib import hist_plotter #not ever directly called, but changes to pyplot are necessary


np.set_printoptions(precision = 3)

imagej_loc = h.get_ImageJ_location(_platform)
ij_macro_loc = os.path.join(g.lib, 'aniso_macro.ijm')
macro_label = 'macro_performed_list.txt'

h.prompt_user_to_set_up_files()


rel_paths, dep_ims_fname_ll = h.set_up_outputs()


#run ImageJ macros to get OrientationJ measures (?)

for rel_path, dep_im_fnames in zip(rel_paths, dep_ims_fname_ll):
    if rel_path == '.': #same as trunk
        rel_path = ''
        
    print("Now working on files in {0}...".format(os.path.join(g.dep,rel_path)))
    
    if len(dep_im_fnames) == 0: # no images to process
        continue    
    
    
    out_dir = os.path.join(g.out_dir, rel_path)    

    
    for dep_im_fname in dep_im_fnames:
        if dep_im_fname.endswith('.tif'):
            if g.ANISO_LABEL in dep_im_fname:
                aniso_fname, aniso_im = h.get_image(g.BATCH, fpath = os.path.join(rel_path, dep_im_fname))
                
                # generate the Text Images if not already created
                im_path = os.path.join(g.dep, rel_path, aniso_fname)
                
                #check to see if the file's been processed
             #   if macro_label not in os.listdir(os.path.join(g.dep,rel_path)): 
                    # 'a+' allows for appending and reading, creates file if non-existent
                with open(os.path.join(g.dep,rel_path,macro_label), mode = 'a+') as inf:
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
                    #suppressing output...
                    #will throw subprocess.CalledProcessError exception if the subprocess fails for some reason...
                    #but if not, hopefully that means the macro also worked correctly
                    #so let's mark that it ran here on a specific file
                    with open(os.path.join(g.dep,rel_path,macro_label), 'a+') as inf:
                        inf.write('{0}\n'.format(im_path))
            
                
            elif g.NSC_LABEL in dep_im_fname:
                dye_fname, dye_im = h.get_image(g.BATCH, fpath = os.path.join(rel_path, dep_im_fname))



    im_data_shape = tuple(reversed(dye_im.size))
    
    f_prefix = dye_fname
    
    #color_of_interest = h.get_color_of_interest()
    color_of_interest = 'brown'
    
    coords_of_interest = h.collect_coords_of_interest(dye_im, color_of_interest)
    
    Z = 4
    
    #anisotropy...
    
    aniso_data = h.get_aniso_data(flag = g.BATCH, relpath = rel_path) #for anisotropy data
    #aniso_fname, aniso_im = h.get_image(g.ANISO_IM)
    
    validity_mask, outlier_coords = h.make_validity_mask(np.array(aniso_im.convert('L')), z = Z)
    
    
    # for when using quantile method of getting outliers...
    #validity_mask = None  #filler
    #outlier_coords = h.get_outliers(np.array(aniso_im.convert('L')))
    
    h.remove_coords(data = aniso_data, coords = outlier_coords)
    bg_coords = h.remove_background(data = aniso_data)
    
    
    coords_of_interest = [c for c in coords_of_interest if c not in outlier_coords and c not in bg_coords]
    
    high_aniso_coords, method_high = h.get_coords(aniso_data, data_mask = validity_mask, predicate = h.has_high_aniso) #eg, in this case, white matter
    low_aniso_coords, method_low = h.get_coords(aniso_data, data_mask = validity_mask, predicate = h.has_low_aniso) #eg, in this case, gray matter
    
    #remove now
    
    
    hist_ftype = 'PNG'
    
    
    #naming histograms...
    coords_list = [coords_of_interest, high_aniso_coords, low_aniso_coords, aniso_data.keys()]
    coord_names = ['dye_coords', 'high_aniso_coords method={0}'.format(method_high), 'low_aniso_coords method={0}'.format(method_low), 'all_coords_(no_bg_or_artifacts)']
    
    #clean coords in coords_list of any outlier or background coordinates...
    
    
    
    for coords, coords_name in zip(coords_list, coord_names):
        n_bins = 100
        hist_info = "dye_imagename={0} color_of_interest={1} {2} (n={3}) histogram (bins={4}).{5}".format(dye_fname, color_of_interest, coords_name, len(coords), n_bins, hist_ftype) #filename
        hist_title = "Coherence-weighted energy values for specified group: {0}".format(coords_name)
        h.plot_histogram_data(data = aniso_data, coords = coords, outdir = out_dir, fname = hist_info, title = hist_title, predicate = h.weighted_anisotropy, bins = n_bins)
    
    
    ## Coordinates of Interest
    #hist_info = 'dye_fname={0} color_of_interest_(HSV)={1} coords_of_interest histogram.{2}'.format(dye_fname, color_of_interest, hist_ftype)
    #hist_title = 'E*C for within specified color threshold.'
    #h.plot_histogram_data(aniso_data, coords = coords_of_interest, fname = hist_info, title = hist_title, predicate = h.weighted_anisotropy, bins = 100)
    
    
    ## All non-artifact/non-background coordinates
    #
    #hist_info = 'dye_fname={0} all_coordinates histogram.{1}'.format(dye_fname, hist_ftype)
    #if not os.path.exists(os.path.join(g.cache_dir, hist_info)):
    #    hist_title = 'E*C for entire image.'
    #    indexer = np.ndindex(im_data_shape)
    #    h.plot_histogram_data(aniso_data, coords = indexer, fname = hist_info, title = hist_title, predicate = h.weighted_anisotropy, bins = 100)
    #    
    
    
    
    #pixels_of_interest = [tuple(reversed(i)) for i in coords_of_interest] #reversed to match (x,y)
    #pixels_info = 'dye_fname={0} color_of_interest_(HSV)={1} pixels_of_interest.p'.format(dye_fname,color_of_interest)
    #h.save_to_cache(pixels_of_interest, pixels_info)
    
    
    h.map_marked_pixels(outpath = out_dir, coords = coords_of_interest, image_shape = dye_im.size, fname = '{0} dye_marked_pixels.png'.format(f_prefix))
    h.map_marked_pixels(outpath = out_dir, coords = bg_coords, image_shape = dye_im.size, fname = '{0} bg_pixels.png'.format(f_prefix)) #debugging
    h.map_marked_pixels(outpath = out_dir, coords = outlier_coords, image_shape = dye_im.size, fname = '{0} outlier_pixels z={1}.png'.format(f_prefix, Z)) #debugging
    
    # TODO: update outpath when batching is implemented

print('Done!')