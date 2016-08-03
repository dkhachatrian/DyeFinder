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

#color_of_interest = h.get_color_of_interest()
color_of_interest = 'brown'

coords_of_interest = h.collect_coords_of_interest(dye_im, color_of_interest)



#anisotropy...

aniso_data = h.get_aniso_data() #for anisotropy data
aniso_fname, aniso_im = h.get_image(h.ANISO_IM)

validity_mask, outlier_coords = h.make_validity_mask(np.array(aniso_im.convert('L')))


h.remove_outliers(data = aniso_data, outlier_coords)
h.remove_background(data = aniso_data)


coords_of_interest = [c for c in coords_of_interest if c not in outlier_coords]

high_aniso_coords, method_high = h.get_coords(aniso_data, data_mask = validity_mask, predicate = h.has_high_aniso) #eg, in this case, white matter
low_aniso_coords, method_low = h.get_coords(aniso_data, data_mask = validity_mask, predicate = h.has_low_aniso) #eg, in this case, gray matter


hist_ftype = 'PNG'

# Coordinates of Interest
hist_info = 'dye_fname={0} color_of_interest_(HSV)={1} coords_of_interest histogram.{2}'.format(dye_fname, color_of_interest, hist_ftype)
hist_title = 'E*C for within specified color threshold.'
h.plot_histogram_data(aniso_data, coords = coords_of_interest, fname = hist_info, title = hist_title, predicate = h.weighted_anisotropy, bins = 100)


# All coordinates

hist_info = 'dye_fname={0} all_coordinates histogram.{1}'.format(dye_fname, hist_ftype)
if not os.path.exists(os.path.join(g.cache_dir, hist_info)):
    hist_title = 'E*C for for entire image.'
    indexer = np.ndindex(im_data_shape)
    h.plot_histogram_data(aniso_data, coords = indexer, fname = hist_info, title = hist_title, predicate = h.weighted_anisotropy, bins = 100)
    



pixels_of_interest = [tuple(reversed(i)) for i in coords_of_interest] #reversed to match (x,y)
pixels_info = 'dye_fname={0} color_of_interest_(HSV)={1} pixels_of_interest.p'.format(dye_fname,color_of_interest)
h.save_to_cache(pixels_of_interest, pixels_info)


h.map_marked_pixels(outpath = g.out_dir, coords = coords_of_interest, image_shape = dye_im.size) #debugging
# TODO: update outpath when batching is implemented

print('Done!')