# -*- coding: utf-8 -*-
"""
Created on Apr 25th 2025
@author: Weiran Wu

Samping in Spatio Omics: 
without state normalization
24 dimensional discrete action, 
penalty for overlapped area
"""

import numpy as np
from gymnasium import spaces
import csv
import random
import os
from graph_build_remake import load_cell_data, construct_graph_for_FOV


class SpatOmics_graph():
    def __init__(self, args, exp):

        ## load raw data from csv
        coords_file_path = os.path.join('dataset_cortex', '{}_spatial.csv'.format(exp))
        cell_types_file_path = os.path.join('dataset_cortex', '{}_subclass.csv'.format(exp))
        biomarker_expression_file_path = os.path.join('dataset_cortex', '{}_expression.csv'.format(exp))

        self.cell_data = load_cell_data(
            cell_coords_file=coords_file_path,
            cell_types_file=cell_types_file_path,
            cell_biomarker_expression_file=biomarker_expression_file_path)

        self.cell_types = sorted(self.cell_data['CELL_TYPE'].unique())
        self.cell_type_num = len(self.cell_types)
        self.cell2sub = dict(zip(self.cell_data['CELL_ID'],
                            self.cell_data['CELL_TYPE']))
        self.sub2idx = {cls: i for i, cls in enumerate(self.cell_types)}

        """Length of expression marker should be added later"""


        # set range
        self.x_min = self.cell_data['X'].min()
        self.x_max = self.cell_data['X'].max()
        self.y_min = self.cell_data['Y'].min()
        self.y_max = self.cell_data['Y'].max()
        # set grid size
        self.grid_width = 50
        self.grid_height = 50

        # count grids
        self.map_x_range = int(np.ceil((self.x_max - self.x_min) / self.grid_width))
        self.map_y_range = int(np.ceil((self.y_max - self.y_min) / self.grid_height))

        # set FOV size
        self.rs = args.rs

        # set target cell
        self.target = 'L45_IT'

        # set spaces
        """There are some issues with how the observation space is defined. 
        The state is returned as an expression matrix with a fixed number of columns 
        (equal to the number of markers or cell types) but a variable number of rows 
        (equal to the number of cells within the FOV). This setup raises several problems—
        —for example, what should the system do if the FOV contains zero cells? """
        self.action_space = spaces.Discrete(8+16)


    def step(self, action, rand=False, isEval=False):
        self.count += 1
        action = action + 1 
        x, y = self.pos_sampling

        if rand:
            next_x, next_y = (round(random.uniform(self.x_min, self.x_max), 1), round(random.uniform(self.y_min, self.y_max), 1))
        else:
            if action < 9:
                if action in [1, 4, 6]:
                    next_x = x - self.rs
                elif action in [2, 7]:
                    next_x = x
                else:
                    next_x = x + self.rs
                if action in [1, 2, 3]:
                    next_y = y + self.rs
                elif action in [4, 5]:
                    next_y = y
                else:
                    next_y = y - self.rs
            else:
                if 9 <= action <= 13:
                    next_x = x + (action-11) *self.rs
                    next_y = y + 2 *self.rs
                elif action in [14, 15, 16]:
                    next_x = x + 2 *self.rs
                    next_y = y + (15-action) *self.rs
                elif action in [17, 18, 19]:
                    next_x = x - 2 *self.rs
                    next_y = y + (18-action) *self.rs
                else:
                    next_x = x + (action-22) *self.rs
                    next_y = y - 2 *self.rs

        # projection to the boundary
        next_x = self.x_min + self.rs/2 if next_x < self.x_min + self.rs/2 else next_x
        next_x = self.x_max - self.rs/2 if next_x > self.x_max - self.rs/2 else next_x
        next_y = self.y_min + self.rs/2 if next_y < self.y_min + self.rs/2 else next_y
        next_y = self.y_max - self.rs/2 if next_y > self.y_max - self.rs/2 else next_y
        self.pos_sampling = [next_x, next_y]

        """With each step, we rebuild a new graph to update the state. Should the environment simply 
        return the graph object, or would it be better to return its raw components—namely, 
        the X matrix and the edge_index array—directly?"""
        FOV_G = construct_graph_for_FOV(
            FOV_center=self.pos_sampling,
            range_radius=self.rs/2,
            cell_data=self.cell_data,
        )


        N = FOV_G.number_of_nodes()
        x_matrix = np.zeros((N, self.cell_type_num), dtype=float)

        for node, data in FOV_G.nodes(data=True):
            cid = data['cell_id']
            sub = self.cell2sub.get(cid, None)
            if sub is not None:
                x_matrix[node, self.sub2idx[sub]] = 1.0

        edge_list = np.array(list(FOV_G.edges()), dtype=int)
        edge_index = edge_list.T  # shape (2, E)


        ## get reward
        # reward for overlapped area
        self.update_map()
        r_overlap = self.get_overlap_area()/1000
        # reward for AD
        AD_counts = self.measure()

        r_AD = np.sum(AD_counts)

        if isEval:
            reward = 5 * r_AD
        else:
            reward = 1*r_overlap + 5*r_AD
            if r_AD > 2:
                 self.success += 1
            if self.success > 9:
                self.done = True
                reward += 100

        self.samp_corner_store.append((next_x-self.rs/2, next_y-self.rs/2, next_x+self.rs/2, next_y+self.rs/2, 0))

        ## get state
        row_sampling = int((next_x - self.x_min) / self.grid_width) - 1
        col_sampling = int((next_y - self.y_min) / self.grid_height) - 1
        abs_map = [1] + self.grid_far(row_sampling, col_sampling) + self.grid_around(row_sampling, col_sampling)

        """To incorporate map information, we first use the GNN to compute its embedding, 
        then concatenate this with the existing embeddings, and finally pass the combined vector 
        through a fully-connected layer. """
        #state = np.concatenate((np.array([next_x, next_y]), np.array(abs_map), mk))

        return x_matrix, edge_index, np.array(abs_map), reward, self.done, r_AD


    def update_map(self):

        x_min_index = max(0, int((self.pos_sampling[0] - self.rs - self.x_min) / self.grid_width))
        x_max_index = min(self.map_x_range-1, int((self.pos_sampling[0] + self.rs - self.x_min) / self.grid_width))
        y_min_index = max(0, int((self.pos_sampling[1] - self.rs - self.y_min) / self.grid_height))
        y_max_index = min(self.map_y_range-1, int((self.pos_sampling[1] + self.rs - self.y_min) / self.grid_height))


        for i in range(x_min_index, x_max_index+1):
            for j in range(y_min_index, y_max_index+1):
                self.map[i][j] += 1   

    def measure(self):
        x_s = self.pos_sampling[0]
        y_s = self.pos_sampling[1]

        # points in range
        cell_data_in_range = self.cell_data[
            (self.cell_data['X'] >= x_s - self.rs/2) &
            (self.cell_data['X'] <= x_s + self.rs/2) &
            (self.cell_data['Y'] >= y_s - self.rs/2) &
            (self.cell_data['Y'] <= y_s + self.rs/2)
            ]

        # count CELL_TYPE
        target_count = (cell_data_in_range['CELL_TYPE'] == self.target).sum()

        return target_count



    def get_overlap_area(self):
        reward = 0
        x_center, y_center = self.pos_sampling[0], self.pos_sampling[1]
        for i, (x1, y1, x2, y2, n) in enumerate(self.samp_corner_store):
            overlap_x1 = max(x_center - self.rs / 2, x1)
            overlap_x2 = min(x_center + self.rs / 2, x2)
            overlap_y1 = max(y_center - self.rs / 2, y1)
            overlap_y2 = min(y_center + self.rs / 2, y2)
            overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
            if overlap_area > 0:
                self.samp_corner_store[i] = (x1, y1, x2, y2, n+1)
            reward += overlap_area * (n+1)
        return -reward

 
    def grid_around(self, row, col):
        around_grids = [0 for _ in range(8)]
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for i, (dr, dc) in enumerate(directions):
            new_row, new_col = row+dr, col+dc
            if 0 <= new_row < self.map_x_range and 0 <= new_col < self.map_y_range:
                around_grids[i] = self.map[new_row][new_col]
            else:
                around_grids[i] = 0
        
        return around_grids
    
    def grid_far(self, row, col):

        average_grids = [0 for _ in range(8)]
        num_grids = [0 for _ in range(8)]
        for i in range(self.map_x_range):
            for j in range(self.map_y_range):
                if i < row-1 and j < col-1:
                    average_grids[0] += self.map[i][j]
                    num_grids[0] += 1
                elif i in [row-1,row,row+1] and j < col-1:
                    average_grids[1] += self.map[i][j]
                    num_grids[1] += 1
                elif i > row+1 and j < col-1:
                    average_grids[2] += self.map[i][j]
                    num_grids[2] += 1
                elif i < row-1 and j in [col-1,col,col+1]:
                    average_grids[3] += self.map[i][j]
                    num_grids[3] += 1
                elif i > row+1 and j in [col-1,col,col+1]:
                    average_grids[4] += self.map[i][j]
                    num_grids[4] += 1
                elif i < row-1 and j > col+1:
                    average_grids[5] += self.map[i][j]
                    num_grids[5] += 1
                elif i in [row-1,row,row+1] and j > col+1:
                    average_grids[6] += self.map[i][j]
                    num_grids[6] += 1
                elif i > row+1 and j > col+1:
                    average_grids[7] += self.map[i][j]
                    num_grids[7] += 1
        for k in range(8):
            average_grids[k] = average_grids[k]/num_grids[k] if num_grids[k]>0 else 0 
        return average_grids


    def reset(self):
        self.count = 0
        self.success = 0
        self.done = False
        self.samp_corner_store = []
        self.map = [[0 for _ in range(self.map_y_range)] for _ in range(self.map_x_range)]
        ## reset the initial sampling position
        x, y = (round(random.uniform(self.x_min, self.x_max), 1),
                round(random.uniform(self.y_min, self.y_max), 1)) # random

        self.pos_sampling = [x, y]

        FOV_G = construct_graph_for_FOV(
            FOV_center=self.pos_sampling,
            range_radius=self.rs/2,
            cell_data=self.cell_data,
        )

        N = FOV_G.number_of_nodes()
        x_matrix = np.zeros((N, self.cell_type_num), dtype=float)

        for node, data in FOV_G.nodes(data=True):
            cid = data['cell_id']
            sub = self.cell2sub.get(cid, None)
            if sub is not None:
                x_matrix[node, self.sub2idx[sub]] = 1.0

        edge_list = np.array(list(FOV_G.edges()), dtype=int)
        edge_index = edge_list.T  # shape (2, E)


        self.samp_corner_store.append((x-self.rs/2, y-self.rs/2, x+self.rs/2, y+self.rs/2, 0))
        self.update_map()

        row_sampling = int((self.pos_sampling[0] - self.x_min) / self.grid_width) - 1
        col_sampling = int((self.pos_sampling[1] - self.y_min) / self.grid_height)- 1
        abs_map = [1] + self.grid_far(row_sampling, col_sampling) + self.grid_around(row_sampling, col_sampling)

        # state = np.concatenate((np.array([x, y]), np.array(abs_map), mk))


        return x_matrix, edge_index, np.array(abs_map)
    
