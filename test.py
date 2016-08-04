# -*- coding: utf-8 -*-

from lib import helpers as h
from lib import globe as g
import numpy as np
import os


aniso_fname, aniso_im = h.get_image(g.ANISO_IM)
aniso_data = h.get_aniso_data() #for anisotropy data


ec_arr = np.ndarray(tuple(reversed(aniso_im.size)))

for k in aniso_data: # unpacks the key ( a tuple )  when I put k,v ...
    v = aniso_data[k]
    ec_arr[k] = v.coherence * v.energy


validity_mask, outlier_coords = h.make_validity_mask(ec_arr)

h.map_marked_pixels(outpath = g.out_dir, coords = outlier_coords, image_shape = aniso_im.size, fname = 'outlier_pixels.png') #debugging

print('Done!')