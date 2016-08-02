# -*- coding: utf-8 -*-

"""

Finds pixels with specified color and coregisters them with anisotropy data. Produces a histogram showing distribution of pixels of interest with anisotropy information at said pixel.

"""


from lib import helpers as h
import numpy as np
import os
#from collections import namedtuple


np.set_printoptions(precision = 3)



dep_ims = h.set_up_outputs()



dye_fname, dye_im = h.get_image(h.DYE_IM)
im_data_shape = tuple(reversed(dye_im.size))
color_of_interest = h.get_color_of_interest()

coords_of_interest = h.collect_coords_of_interest(dye_im, color_of_interest)


#aniso_fname, aniso_im = h.get_image(h.ANISO_IM)

tupled_data = h.get_data() #for anisotropy data

hist_ftype = 'PNG'

# Coordinates of Interest
hist_info = 'dye_fname={0} color_of_interest_(HSV)={1} coords_of_interest histogram.{2}'.format(dye_fname, color_of_interest, hist_ftype)
hist_title = 'E*C for within specified color threshold.'
h.plot_histogram_data(tupled_data, coords = coords_of_interest, fname = hist_info, title = hist_title, predicate = h.weighted_anisotropy, bins = 100)


# All coordinates

hist_info = 'dye_fname={0} all_coordinates histogram.{1}'.format(dye_fname, hist_ftype)
if not os.path.exists(os.path.join(h.cache_dir, hist_info)):
    hist_title = 'E*C for for entire image.'
    indexer = np.ndindex(im_data_shape)
    h.plot_histogram_data(tupled_data, coords = indexer, fname = hist_info, title = hist_title, predicate = h.weighted_anisotropy, bins = 100)
    



pixels_of_interest = [tuple(reversed(i)) for i in coords_of_interest] #reversed to match (x,y)
pixels_info = 'dye_fname={0} color_of_interest_(HSV)={1} pixels_of_interest.p'.format(dye_fname,color_of_interest)
h.save_to_cache(pixels_of_interest, pixels_info)



print('Done!')