# -*- coding: utf-8 -*-

"""

Class for reading in values from a file whose format matches that output by
ImageJ's "Save As...Text Image" option.

"""







class TextImageReader:
    """
    A class to access data housed in files whose structure mimicks that of
    the "Save As...Text Image" option in ImageJ.
    
    Useful for very high-resolution images where memory may be a concern, and
    when saving manipulations performed on the Text Image-like file needn't
    be performed.
    (Or else, if necessary, could make a copy of the original file and perform
    operations on them.)
    
    Motivation for this class: the resulting dictionaries (as measured by the size
    of the pickled object relative to the original image) formed from running
    DyeFinder on a 1586x1113 ~5 MB image were ~70 MB.
    
    So for the higher resolution ~700 MB images, would require ~10 GB memory
    to hold dictionary...
    
    Note: using a TextImageReader and not a dictionary means that lists of coordinates
    will be needed to locate specific values.
    """
    
    def __init__(self, fpath, delimiter = '\t'):
        self.path = fpath
        self.delimiter = delimiter
        self.line_offsets = self._parse_offsets() #quicker access to later lines
        self.dshape = self._get_shape()
#        pass
    
    def __getitem__(self, key):
        """
        Allows indexing similar to a list/NumPy array.
        key must be int or 2-tuple
        """
        if type(key) is int:
            pass #lookup key'th value, going across row then down a column
        elif type(key) is tuple or type(key) is list:
            if len(key) == 1:
                pass #act as if it's an int
            elif len(key) == 2:
                pass #lookup value at key like coordinates
        else:
            pass #can't handle!
        pass
    
    
    def _parse_offsets(self):
        """
        Get byte offsets for start of each line.
        (Ideally will aid in value lookup.)
        """
        # adapted from http://stackoverflow.com/a/620492
        offsets = []
        offset = 0
        with open(self.path, mode = 'r') as inf:
            for line in inf:
                offsets.append(offset)
                offset += len(line)
        
        return offsets
    
    def _get_shape(self):
        """
        Figure out dimensions (in data-structure order) of file.
        """
        # first dimension is number of lines
        # so length of line_offsets
        d1 = len(self.line_offsets)
        
        # second dimension assumed to be constant
        # use first line and tab spacing
        with open(self.path, mode = 'r') as inf:
            line = inf.readline()
            d2 = 1 + len(line.split(self.delimiter))
        
        # return as tuple
        return (d1,d2)
        
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        # TODO: format variables...
        pass
    
#    __str__ = __repr__
    
    
    # probably could be useful
    # from http://stackoverflow.com/a/620492
    
## Read in the file once and build a list of line offsets
#line_offset = []
#offset = 0
#for line in file:
#    line_offset.append(offset)
#    offset += len(line)
#file.seek(0)
#
## Now, to skip to line n (with the first line being line 0), just do
#file.seek(line_offset[n])