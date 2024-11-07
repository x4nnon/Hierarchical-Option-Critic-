#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:22:27 2023

@author: x4nno
"""

import numpy as np
import pickle 
import matplotlib.pyplot as plt

def cluster_PI_compression(clusterer, concat_fractures, all_s_f, all_ts,
                           chain_length=2, max_cluster_returns=100, 
                           min_PI_score = 0.05):
    
    # cluster count successes and failures 
    cluster_success_dict = {}
    cluster_failure_dict = {}
    
    print("number of clusters are ")
    print(clusterer.labels_.max())
    
    for i in range(len(clusterer.labels_)):
        if clusterer.labels_[i] != -1:  # because -1 is an outlier and not a cluster
            if clusterer.labels_[i] not in cluster_success_dict.keys():
                cluster_success_dict[clusterer.labels_[i]] = 1 # init as 1
            if clusterer.labels_[i] not in cluster_failure_dict.keys():
                cluster_failure_dict[clusterer.labels_[i]] = 1 # init as 1

            if all_s_f[i] > 0:
                cluster_success_dict[clusterer.labels_[i]] += 1
            else:
                cluster_failure_dict[clusterer.labels_[i]] += 1
                
    # find success probability 
    
    success_probs = {}
    for cluster in cluster_success_dict.keys():
        failure_probs = cluster_failure_dict[cluster] /         \
            (cluster_success_dict[cluster] + cluster_failure_dict[cluster])
        
        success_probs[cluster] = (cluster_success_dict[cluster] /    \
            (cluster_success_dict[cluster] + cluster_failure_dict[cluster])) -   \
                failure_probs

        ######## We have added the failure probability as a negative here.

    # find choice probability 
    
    choice_probs = {}
    ### Because the failure trajectories create many more fractures / clusters 
    ### we want total number of choices over success and failure seperately.
    
    total_number_success_choices = 1
    total_number_failure_choices = 1
    for cluster in clusterer.labels_:
        if cluster != -1:
            total_number_success_choices += (cluster_success_dict[cluster] -1 )
            total_number_failure_choices += (cluster_failure_dict[cluster] - 1)
    
    for cluster in cluster_success_dict.keys():
        choice_probs[cluster] = ((cluster_success_dict[cluster] -1) /             \
            (total_number_success_choices+1) - ((cluster_failure_dict[cluster]-1) / (total_number_failure_choices+1)))
            
    # we need to find our entropy term:
        
    # sum(P(z,x))*np.log(p(z,x)/p(x))
    
    task_action_count = []
    cumu_task_action_count = []
    cumu_count = 0
    task_cluster_dict = {}
    task_count = 0
    for traj in all_ts:
        task_action_count.append(len(traj)-chain_length)
        cumu_count += len(traj)-chain_length
        cumu_task_action_count.append(cumu_count)
        task_cluster_dict[task_count] = {}
        task_count += 1
        
    entropy_dict = {}
    
    for i in range(len(clusterer.labels_)):
        if clusterer.labels_[i] != -1:  # because -1 is an outlier and not a cluster                
            for t1 in range(task_count): # init the dict
                if clusterer.labels_[i] not in task_cluster_dict[t1].keys():
                    task_cluster_dict[t1][clusterer.labels_[i]] = 1
                if clusterer.labels_[i] not in entropy_dict.keys():
                    entropy_dict[clusterer.labels_[i]] = 0 # why was this 1 before!?
            if all_s_f[i] > 0:
                task_idx = 0
                for task_len in cumu_task_action_count:
                    if i < task_len:
                        task_cluster_dict[task_idx][clusterer.labels_[i]] += 1
                        break
                    else:
                        pass # here just for debug
                    task_idx += 1
            # else:
            #     task_idx = 0
            #     for task_len in cumu_task_action_count:
            #         if i < task_len:
            #             task_cluster_dict[task_idx][clusterer.labels_[i]] -= 1
            #             break
            #         else:
            #             pass # here just for debug
            #         task_idx += 1
    
    
    unique_clusters = list(set(clusterer.labels_))
    
    
    for cluster in unique_clusters:
        if cluster != -1:
            for task_idx in task_cluster_dict.keys():
                try:
                    P_zgx = task_cluster_dict[task_idx][cluster] / (task_action_count[task_idx])
                except: # Sometimes the trajectories are lower or the same as the chain_length this stops div by 0.
                    P_zgx = task_cluster_dict[task_idx][cluster] / 1
                # P_x = 1/task_count
                # print("P_x : ", P_x)
                entropy_adder = -(P_zgx*np.emath.logn(len(unique_clusters), P_zgx))
                entropy_dict[cluster] += entropy_adder
            
            entropy_dict[cluster] = entropy_dict[cluster]/task_count # 1*task_count is perfect so weighted here 
            
            
            
        # Require this final division to bring it inline with the rest of our calcs
        
    # 3) Find highest scoring cluster
    
    # print(entropy_dict)
    
    top_cluster = None
    top_PI_score = 0
    all_success_clusters = []
    cluster_pi_dict = {}
    for cluster in success_probs.keys():
        if cluster_success_dict[cluster] > 1:
            # print("a success cluster is ", cluster)
            all_success_clusters.append(cluster)
        PI_score = (success_probs[cluster] + choice_probs[cluster] \
            + entropy_dict[cluster])/3 # 3 is the best score we can get therefore results is percent
            # + \
            # np.log10(get_BIC(cluster_success_dict, cluster))
        cluster_pi_dict[cluster] = PI_score
        if (PI_score > top_PI_score):
            top_PI_score = PI_score
            top_cluster = cluster
        
    ordered_cluster_pi_dict = dict(sorted(cluster_pi_dict.items(), key=lambda item: item[1], reverse=True))
    
    clus_count = 0
    best_clusters_list = []
    for clus in ordered_cluster_pi_dict.keys():
        # print("clus is ", clus, "PI SCORE IS :", ordered_cluster_pi_dict[clus])
        # print(" fail is ", cluster_failure_dict[clus])
        # print(" success is ", cluster_success_dict[clus])
        # print(clus, entropy_dict[clus], success_probs[clus], choice_probs[clus])
        if (clus_count < max_cluster_returns) & (ordered_cluster_pi_dict[clus] > min_PI_score):
            # This 0.05 limit is arbitary
            best_clusters_list.append(clus)
        clus_count += 1
    # can use the ordered cluster_pi_dict to return only the top x 
    # amount according the the PI compression: although can just not do it..
    
    return clusterer, top_cluster, all_success_clusters,\
        ordered_cluster_pi_dict, best_clusters_list
        