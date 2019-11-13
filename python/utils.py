'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name


def set_diff(a, b):
    '''Return the set difference of the two arrays.'''
    return list(set(a).difference(set(b)))
