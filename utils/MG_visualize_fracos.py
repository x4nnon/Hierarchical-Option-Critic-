#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:07:36 2024

@author: x4nno
"""
import sys 
sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv")
sys.path.append("/home/x4nno_desktop/Documents/MetaGridEnv/MetaGridEnv")

from Environment_obstacles import Environment

import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from matplotlib.patches import Circle, FancyArrowPatch, ArrowStyle
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# we can only really accurately visualize the first depth here?
directions = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0)
}

def calculate_path_hash(coords, actions):
    return hash(tuple(coords) + tuple(actions))

def find_center_of_match(sub_array, large_array):
    sub_shape = sub_array.shape
    large_shape = large_array.shape
    match_coordinates = []

    # Define the offset for the center of the sub-array
    center_offset = (sub_shape[0] // 2, sub_shape[1] // 2)

    # Slide over the large array
    for i in range(large_shape[0] - sub_shape[0] + 1):
        for j in range(large_shape[1] - sub_shape[1] + 1):
            # Extract the sub-array from the large array
            if np.array_equal(large_array[i:i+sub_shape[0], j:j+sub_shape[1]], sub_array):
                # Compute the coordinates of the center of the matching sub-array
                center_coordinates = (i + center_offset[0], j + center_offset[1])
                match_coordinates.append(center_coordinates)

    return match_coordinates


def show_all_concat_fractures(concat_fractures, cluster_reverse_cypher):
    for cf in concat_fractures:
        reshaped = np.array(cf[:49]).reshape((7,7))
        reshaped[3][3] = 2
        plt.imshow(reshaped)
        plt.title(f"{cluster_reverse_cypher[tuple(cf[51:55])]}, {cluster_reverse_cypher[tuple(cf[55:])]}")
        plt.show()


def visualize_fracos_in_const_state(fracos_clusters_dir, depth, chain_length, env_domain, 
                                    number_clusters, domain_name):
    # depth of 0 would be used for the first set of visualizing the options right ... 
    
    clusterers = [pickle.load(open(fracos_clusters_dir+"/clusterers/{}".format(file), "rb")) \
                  for file in sorted(os.listdir(fracos_clusters_dir+"/clusterers"))]
        
    clusters_list = [pickle.load(open(fracos_clusters_dir+"/clusters/{}".format(file), "rb")) \
                  for file in sorted(os.listdir(fracos_clusters_dir+"/clusters"))]
        
    
        
    concat_fractures_list = [pickle.load(open(fracos_clusters_dir+"/other/{}".format(file), "rb")) \
                  for file in sorted(os.listdir(fracos_clusters_dir+"/other")) if "concat_fractures" in file]

    cluster_reverse_cyphers_list = [pickle.load(open(fracos_clusters_dir+"/cluster_reverse_cyphers/{}".format(file), "rb")) \
                  for file in sorted(os.listdir(fracos_clusters_dir+"/cluster_reverse_cyphers"))]
        
    clusterer = clusterers[depth]
    clusters = clusters_list[depth]
    clusters_list[depth]
    clusters = clusters[:number_clusters]
    concat_fractures = concat_fractures_list[depth]
    cluster_reverse_cypher = cluster_reverse_cyphers_list[depth]
    
    # show_all_concat_fractures(concat_fractures, cluster_reverse_cypher)
    
    obs_list = [cf[:49] for cf in concat_fractures]
    action_list = [cf[51:] for cf in concat_fractures]
    
    print("debug point 1")
    
    rev_cyph_action_list = [[cluster_reverse_cypher[tuple(action_seq[int(i*len(action_seq)/chain_length):int((i+1)*len(action_seq)/chain_length)])] for i in range(chain_length)]for action_seq in action_list]

    print("debug point 2")
    
    # now for each obs we turn to a 7x7 and find centre of match,
    
    co_ord_list = []
    for obs in obs_list:
        obs = np.array(obs).reshape((7,7))
        co_ords = find_center_of_match(obs, env_domain)
        co_ord_list.append(co_ords)
        
    print("debug point 3")
    
    # we now have co_ord_list, action_list and clusterer.labels_ where each of the indexes match up
    # so we need to loop over each clusterer label, find all indexes of this label in clusterer,labels_
    # then we can plot our arrays from the corresponding
    
    
    path_frequency = defaultdict(int)

    # Calculate frequencies of each path
    for label in np.unique(clusterer.labels_):
        if label in clusters:
            label_indices = np.where(clusterer.labels_ == label)
            for lab_ind in label_indices[0]:
                all_coords = []
                all_actions = []
                for co_ord in co_ord_list[lab_ind]:  # there are potentially multiple co-ords here
                    path_coords = []
                    path_actions = []
                    dx = 0
                    dy = 0
                    x = co_ord[1]
                    y = co_ord[0]
                    for act in rev_cyph_action_list[lab_ind]:
                        x += dx
                        y += dy
                        path_coords.append((x, y))
                        dx, dy = directions[act]
                        path_actions.append((dx, dy))
                    all_coords.append(path_coords)
                    all_actions.append(path_actions)
                path_hash = calculate_path_hash(tuple(map(tuple, all_coords)), tuple(map(tuple, all_actions)))
                path_frequency[path_hash] += 1
    
    # Plotting the paths with enhancements
    os.makedirs(f"utils/fracos_vizs/{domain_name}", exist_ok=True)
    for label in np.unique(clusterer.labels_):
        if label in clusters:
            fig, ax = plt.subplots()
            invert_colours = 1 - env_domain
            ax.matshow(invert_colours, cmap="gray")
            ax.set_title(f'Label: {label}')
    
            label_indices = np.where(clusterer.labels_ == label)
            for lab_ind in label_indices[0]:
                all_coords = []
                all_actions = []
                for co_ord in co_ord_list[lab_ind]:  # there are potentially multiple co-ords here
                    path_coords = []
                    path_actions = []
                    dx = 0
                    dy = 0
                    x = co_ord[1]
                    y = co_ord[0]
                    for act in rev_cyph_action_list[lab_ind]:
                        x += dx
                        y += dy
                        path_coords.append((x, y))
                        dx, dy = directions[act]
                        path_actions.append((dx, dy))
                    all_coords.append(path_coords)
                    all_actions.append(path_actions)
                path_hash = calculate_path_hash(tuple(map(tuple, all_coords)), tuple(map(tuple, all_actions)))
                path_freq = path_frequency[path_hash]
    
                # norm = Normalize(vmin=min(path_frequency.values()), vmax=max(path_frequency.values()))
                # colormap = matplotlib.colormaps['Blues']    
    
                # Plot the path
                arrow_style = ArrowStyle("-|>", head_length=1, head_width=1)
                for path_coords, path_actions in zip(all_coords, all_actions):
                    for i, (x, y) in enumerate(path_coords):
                        if i == 0:
                            start_marker = Circle((x, y), 0.15, color='green')  # Start point in green
                            ax.add_patch(start_marker)  # Start point in green
                        elif i == len(path_coords)-1:
                            dx, dy = path_actions[i]
                            end_marker = Circle((x+dx, y+dy), 0.1, color='red')  # End point in red
                            ax.add_patch(end_marker)  # End point in red
                            
                        if i < len(path_coords):
                            dx, dy = path_actions[i]
                            arrow_scale = 1.2 # to stop overlap
                            # color = colormap(norm(path_freq))
                            arrow = FancyArrowPatch((x, y), (x + dx * arrow_scale, y + dy * arrow_scale),
                                            arrowstyle=arrow_style, color="blue", zorder=3, 
                                            linewidth=0.2 + 0.6 * (path_freq / max(path_frequency.values())))
                            ax.add_patch(arrow)
                            # ax.arrow(x, y, dx*arrow_scale, dy*arrow_scale, head_width=0.1, head_length=0.1, fc='blue', ec='blue', 
                            #           linewidth=0.5 + 2 * (path_freq / max(path_frequency.values())))  # Adjust linewidth based on frequency
            # plt.colorbar(ax=ax, label='Frequency')
            plt.savefig("utils/fracos_vizs/{}/{}.svg".format(domain_name, label), format="svg", bbox_inches="tight")
            plt.show()
    
    # for label in np.unique(clusterer.labels_):
    #     if label in clusters:
    #         fig, ax = plt.subplots()
    #         invert_colours = 1-env_domain
    #         ax.matshow(invert_colours, cmap="gray") 
    #         ax.set_title(label)
    #         label_indices = np.where(clusterer.labels_ == label)
    #         for lab_ind in label_indices[0]:
    #             for co_ord in co_ord_list[lab_ind]: # there are potentially multiple co-ords here
    #                 dx = 0
    #                 dy = 0
    #                 for act in rev_cyph_action_list[lab_ind]:
    #                     x = co_ord[1]+dx
    #                     y = co_ord[0]+dy
    #                     dx, dy = directions[act]
    #                     ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2, fc='red', ec='red')
    #         plt.show()
                        

if __name__ == "__main__":
    domain_name = "Four_rooms"
    env1 = Environment(2, 3, domain_size=[14, 14], style=domain_name, seed=42)
    domain = env1.domain
    
    domain[domain==2] = 0
    domain[domain==3] = 0
    
    domain = np.pad(domain, pad_width=3, mode='constant', constant_values=1)

    colors = ["white", "black", "blue", "yellow", "brown"]
    levels = [0, 1, 2, 3, 4, 5]
    
    cm, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
    
    plt.imshow(domain, cmap=cm, norm=norm, interpolation="none")
    plt.show()
    
    visualize_fracos_in_const_state("fracos_clusters/MetaGridEnv/metagrid-v0/{}".format(domain_name), 
                                    0, 4, domain, 10, domain_name)