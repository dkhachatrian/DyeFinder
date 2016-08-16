# -*- coding: utf-8 -*-

"""
Tests outlier choices, based on helpers.make_validity_mask
"""


from lib import helpers as h
from lib import globe as g
import numpy as np
import os

def test_outlier_function():

    aniso_fname, aniso_im = h.get_image(g.ANISO_IM)
    aniso_data = h.get_aniso_data() #for anisotropy data
    
    
    ec_arr = np.ndarray(tuple(reversed(aniso_im.size)))
    
    for k in aniso_data: # unpacks the key ( a tuple )  when I put k,v ...
        v = aniso_data[k]
        ec_arr[k] = v.coherence * v.energy
    
    Z = 4
    
    validity_mask, outlier_coords = h.make_validity_mask(ec_arr, z = Z)
    
    h.map_marked_pixels(outpath = g.out_dir, coords = outlier_coords, image_shape = aniso_im.size, fname = '{0} outlier_pixels z={1}.png'.format(aniso_fname, Z)) #debugging
    
    print('Done!')
    
    

def test_txt_feeder_function():
    path = '/Users/mathoncuser/Dropbox/CoH_Stuff_Summer_2016/Python Scripts/DyeFinder/dependencies/testing_batcher/test ROI_info.txt'
    
    out = h.get_ROI_info_from_txt(path)
    
    return out

def test_load_all_vals():
    path = '/Users/mathoncuser/Dropbox/CoH_Stuff_Summer_2016/Python Scripts/DyeFinder/cache/t = 9 months'
    
    labels2allvals = h.load_all_vals(cache_dir = path, cache_flag = True)
    
    out_dir = os.path.join(g.out_dir, 'testing', g.aggregate_label)
    for label in labels2allvals:
        vals = labels2allvals[label]
        n_bins = 100
        hist_info = ['aggregate', 'brown', label, len(vals), g.hist_ftype]
        #hist_info = "dye_imagename={0} color_of_interest={1} {2} (n={3}) histogram (bins={4}).{5}".format(dye_fname, color_of_interest, coords_name, len(coords), n_bins, hist_ftype) #filename
        title_p = "for specified group: {0}".format(label)
        
        h.plot_histogram_data(vals_dict = labels2allvals, outdir = out_dir, info = hist_info, title_postfix = title_p, bins = n_bins)


#test_txt_feeder_function()    
#test_outlier_function()
test_load_all_vals()
print('Done!')