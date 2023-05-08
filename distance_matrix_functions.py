# import modules
from __future__ import division, print_function
import os
import numpy as np

from itertools import combinations
from pyclesperanto_prototype import rotate
from skimage.draw import ellipsoid
from skimage.segmentation import find_boundaries

def find_all_index_tuples(n_objects, tuple_size = 2):
    
    '''returns all possible combinations of size = tuple_size for n_objects'''
    
    from itertools import combinations
    
    index_tuples = tuple([(a,b) for a,b in combinations(np.arange(n_objects), tuple_size)])
    
    return index_tuples

###-------------------------------------------------------------------------------------------------

def determine_distance_between_two_closed_boundaries(boundary_set1, boundary_set2):
    
    '''returns a raveled distance matrix.
    
    boundary_set1 and boundary_set2 are both 2 dimensional arrays containing coordinates.
    
    boundary_set1 must have size (n1, m).
    boundary_set2 must have size (n2, m).'''
    
    from scipy.spatial.distance import cdist
    
    distance_matrix = cdist(boundary_set1, boundary_set2)
    
    return distance_matrix

###-------------------------------------------------------------------------------------------------

def find_minimum_distance_and_points(distance_matrix):
    
    '''takes a raveled distance matrix'''
    
    minimum_distance = round(np.amin(distance_matrix), ndigits=3)
    min_dist_point_indices = np.unravel_index(np.argmin(distance_matrix, 
                                                        axis=None), 
                                              distance_matrix.shape)
    
    
    return minimum_distance, min_dist_point_indices

###-------------------------------------------------------------------------------------------------

def create_synthetic_dataset(empty_image, n_nuclei):
    
    '''empty_image should be of size (n,n,n) where n>20.'''
    
    max_image_index = empty_image.shape[0] - 1
    
    centroids = np.random.randint(5, high=int(max(empty_image.shape))-5, size=(n_nuclei,len(empty_image.shape)))
    
    axis_a = np.random.randint(12, high=16, size= n_nuclei)
    axis_b = np.random.randint(8, high=12, size= n_nuclei)
    axis_c = np.random.randint(4, high=8, size= n_nuclei)
    
    rotate_a = np.random.randint(0, high=360, size= n_nuclei)
    rotate_b = np.random.randint(0, high=360, size= n_nuclei)
    rotate_c = np.random.randint(0, high=360, size= n_nuclei)
    
    counter = 1
    for i in range(n_nuclei):
        
        nucleus = ellipsoid(axis_a[i], axis_b[i], axis_c[i])
        nucleus = np.pad(nucleus, pad_width=axis_a[i])
        nucleus = rotate(nucleus,
                         rotate_around_center=True,
                         angle_around_x_in_degrees=rotate_a[i], 
                         angle_around_y_in_degrees=rotate_b[i], 
                         angle_around_z_in_degrees=rotate_c[i]).astype(int)
        #imshow(nucleus.sum(axis = 0))
        #plt.show()
        
        #print(nucleus.shape)
        
        half_shape = np.asarray(nucleus.shape) * 0.5
        nuc_floor = np.floor(half_shape).astype(int)
        nuc_ceiling = (np.asarray(nucleus.shape) - nuc_floor).astype(int)
        
        
        d0_l, d1_l, d2_l = centroids[i] - nuc_floor
        d0_h, d1_h, d2_h = centroids[i] + nuc_ceiling
        
        #print((d0_l, d1_l, d2_l), (d0_h, d1_h, d2_h))
        
        if np.sum(empty_image[d0_l:d0_h, d1_l:d1_h, d2_l:d2_h]) > 0:
            continue
        else:
            if (d0_l > 0) & (d1_l > 0) & (d2_l >0) & (d0_h < max_image_index) & (d1_h < max_image_index) & (d2_h < max_image_index):
                nucleus = nucleus * counter
                counter += 1
                #print(empty_image[d0_l:d0_h, d1_l:d1_h, d2_l:d2_h].shape)
                empty_image[d0_l:d0_h, d1_l:d1_h, d2_l:d2_h] = empty_image[d0_l:d0_h, d1_l:d1_h, d2_l:d2_h] + np.ascontiguousarray(nucleus.astype(int))
    
    return empty_image