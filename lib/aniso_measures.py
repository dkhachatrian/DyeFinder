# -*- coding: utf-8 -*-

"""
Various Python functions that can be used to measure the anisotropy
at a given location in an image.
Each function takes in a list of coordinates and its corresponding data (dict),
and returns a corresponding list of values.
Lists are chosen as the container to facilitate histogram plotting with
numpy.histogramdd
(It is expected that the same list of keys (coords) will be used in
simulataneous calls of functions in this module, so the lists will be in the
proper order.)
Coords should already have been  groomed such that all of its elements
are keys in data.
Each function also returns a string (to be used in the histogram title)
to explain the method of anisotropy.

@author: David G. Khachatrian
"""

#from lib import globe as g
#import numpy as np



    
def weighted_anisotropy(data, coords):
    """
    Returns coherence-weighted energy values in a dict of coord-->value.
    Also returns descriptor string.
    """
    label = 'coherence-weighted energy'
    vals = []
    for c in coords:
        vals.append(data[c].coherence * data[c].energy)
    
    return vals, label
    
    #return aniso_tuple[g.label2val['coherence']] * aniso_tuple[g.label2val['energy']]

    
def coherence(data, coords):
    """
    Returns coherence values in a dict of coord-->value. 
    Also returns descriptor string.
    """
    label = 'coherence'
    vals = []
    for c in coords:
        vals.append(data[c].coherence)
    
    return vals, label
    
    
def energy(data, coords):
    """
    Returns coherence values in a dict of coord-->value.
    Also returns descriptor string.
    """
    label = 'energy'
    vals = []
    for c in coords:
        vals.append(data[c].energy)
    
    return vals, label