# import modules
from __future__ import division, print_function
import os
import numpy as np
import networkx as nx

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

###-------------------------------------------------------------------------------------------------

def generate_complete_distance_network(n_objects, index_pairs, minimum_distances):
    
    '''index pairs should be a 2d array'''
    
    graph = nx.Graph()
    
    nodes = np.arange(n_objects)
    graph.add_nodes_from(nodes)
    
    
    reshaped_minimum_distances = np.reshape(minimum_distances, (minimum_distances.shape[0], 1))
    weighted_edges = np.concatenate((index_pairs, reshaped_minimum_distances), axis = 1)
    graph.add_weighted_edges_from([(weighted_edges[i,:]) for i in range(weighted_edges.shape[0])])
    
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    return graph

###-------------------------------------------------------------------------------------------------

def prune_distance_network_by_threshold(graph, threshold):
    
    '''remove all edges larger than threshold'''
    
    pruned_graph = graph.copy()
    
    for edge in graph.edges(data=True):
        for key, value in edge[2].items():
            if key == 'weight':
                if value > threshold:
                    pruned_graph.remove_edge(edge[0], edge[1])
            else:
                continue
    
    return pruned_graph

###-------------------------------------------------------------------------------------------------

def list_network_connected_component_nodes(graph):
    
    '''list nodes of each connected component in a network'''
    
    connected_components_sets = list(nx.connected_components(graph))
    connected_components_lists = [0] * len(connected_components_sets)
    
    counter = 0
    for i in connected_components_sets:
        connected_components_lists[counter] = sorted(list(map(int, i)))
        counter += 1
    
    return connected_components_lists

###-------------------------------------------------------------------------------------------------

def find_network_giant_component(connected_components, connected_components_sizes, network_mean_degree):
    
    qualification_value = network_mean_degree**2 - 2*network_mean_degree
    
    if qualification_value > 0:
        giant_component = connected_components[np.argmax(connected_components_sizes)]
    else:
        giant_component = []
    
    return giant_component

###-------------------------------------------------------------------------------------------------

def calculate_network_density(graph):
    
    n_edges = len(graph.edges())
    n_nodes = len(graph.nodes())
    
    density = (2 * n_edges) / (n_nodes * (n_nodes - 1))
    
    return density
