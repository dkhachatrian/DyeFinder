# -*- coding: utf-8 -*-

"""

Finds pixels with specified color and coregisters them with anisotropy data. Produces a histogram showing distribution of pixels of interest with anisotropy information at said pixel.

"""


from lib import helpers as h
from lib import globe as g
import numpy as np
import os
#from collections import namedtuple


np.set_printoptions(precision = 3)


dep_ims = h.set_up_outputs() #also runs ImageJ macros to get OrientationJ measures



dye_fname, dye_im = h.get_image(g.DYE_IM)
im_data_shape = tuple(reversed(dye_im.size))

f_prefix = dye_fname

#color_of_interest = h.get_color_of_interest()
color_of_interest = 'brown'

coords_of_interest = h.collect_coords_of_interest(dye_im, color_of_interest)

Z = 4

#anisotropy...

aniso_data = h.get_aniso_data() #for anisotropy data
aniso_fname, aniso_im = h.get_image(g.ANISO_IM)

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
    h.plot_histogram_data(data = aniso_data, coords = coords, fname = hist_info, title = hist_title, predicate = h.weighted_anisotropy, bins = n_bins)


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


h.map_marked_pixels(outpath = g.out_dir, coords = coords_of_interest, image_shape = dye_im.size, fname = '{0} dye_marked_pixels.png'.format(f_prefix))
h.map_marked_pixels(outpath = g.out_dir, coords = bg_coords, image_shape = dye_im.size, fname = '{0} bg_pixels.png'.format(f_prefix)) #debugging
h.map_marked_pixels(outpath = g.out_dir, coords = outlier_coords, image_shape = dye_im.size, fname = '{0} outlier_pixels z={1}.png'.format(f_prefix, Z)) #debugging

# TODO: update outpath when batching is implemented

print('Done!')