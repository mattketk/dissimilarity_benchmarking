# import numpy as np
import cupy as np
from pandas import DataFrame
import os
import sys
from time import time

def simulate_dis_mat(n_trees, n_data, n_leaves_per_tree=500, chance_synth=0.4):
    # create a matrix of integers representing the id
    # of each tree leaf
    apply_mat = np.random.randint(0, n_leaves_per_tree, size=(n_data, n_trees)).astype('int16')
    chance_mat = np.random.rand(*apply_mat.shape)
    apply_mat[chance_mat < chance_synth] = -1

    try:
        t0 = time()
        sim_tensor = (apply_mat[:, None] == apply_mat[None, :]) & (apply_mat[:, None] != -1) & (apply_mat[None, :] != -1)
        sim_mat = np.sum(sim_tensor, axis=2) / np.asfarray(np.sum(apply_mat != -1, axis=2), dtype='float')
        t1 = time()
        sim_tensor_mem = sim_tensor.nbytes / 1e9
        time_elapsed = t1 - t0
        print(f'n_trees = {n_trees}, n_data = {n_data}, sim_tensor_mem = {sim_tensor_mem:.2f} GB, time_elapsed = {time_elapsed:.2f} s')
    except MemoryError:
        sim_tensor_mem = -1
        time_elapsed = -1
        print(f'Memory Error: sim_tensor_mem = {sim_tensor_mem:.2f} GB')

    return sim_tensor_mem, time_elapsed

if __name__ == '__main__':
    memory_list = []
    time_list = []
    n_trees_space = [10, 100, 500, 1000]
    n_data_space = [100, 1000, 10000, 100000, 200000, 500000]
    output_dir = '/scratch/gpfs/mk0566/'

    for n_trees in n_trees_space:
        for n_data in n_data_space:
            sim_tensor_mem, time_elapsed = simulate_dis_mat(n_trees, n_data)
            memory_list.append(sim_tensor_mem)
            time_list.append(time_elapsed)
    
    results_frame = DataFrame({'n_trees': n_trees_space * 6,
                                 'n_data': n_data_space * 4,
                                 'memory_GB': memory_list,
                                 'time_elapsed': time_list})
    
    results_frame.to_csv(os.path.join(output_dir, 'benchmarking_results.csv'), index=False)
