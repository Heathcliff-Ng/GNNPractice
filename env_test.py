# -*- coding: utf-8 -*-
"""
Created on Dec 4 2024
@author: Weiran Wu

Samping in Spatio Omics:
without state normalization
24 dimensional discrete action
"""

import numpy as np
from gymnasium import spaces
import csv
import random
import os
from graph_build_remake import load_cell_data, construct_graph_for_FOV


class SpatOmics_graph():
    def __init__(self, args):


        self.cell_data = load_cell_data(
            cell_coords_file=args.cell_coords_file,
            cell_types_file=args.cell_types_file,
            cell_biomarker_expression_file=args.cell_biomarker_expression_file,
            cell_features_file=args.cell_features_file)

        self.fig_save_root = args.fig_save_root

        # set range
        self.x_min = self.cell_data['X'].min()
        self.x_max = self.cell_data['X'].max()
        self.y_min = self.cell_data['Y'].min()
        self.y_max = self.cell_data['Y'].max()

        # set FOV size
        self.rs = args.rs

        # set target cell
        self.target = 'Tumor (CD20+)'


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


        voronoi_polygon_img_output = os.path.join(self.fig_save_root, f"voronoi_step_{self.count}.png")
        graph_img_output = os.path.join(self.fig_save_root, f"graph_step_{self.count}.png")


        state = construct_graph_for_FOV(
            FOV_center=self.pos_sampling,
            range_radius=self.rs/2,
            cell_data=self.cell_data,
            voronoi_polygon_img_output=voronoi_polygon_img_output,
            graph_img_output=graph_img_output
        )

        # get reward
        r_AD = self.measure()

        """
        POSSIBLE OPERATIONS
        POSSIBLE OPERATIONS
        POSSIBLE OPERATIONS
        POSSIBLE OPERATIONS
        """
        reward = r_AD

        print(self.pos_sampling)
        print(reward)

        return state, reward


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


    def reset(self):
        self.count = 0
        self.success = 0
        self.done = False


        ## reset the initial sampling position
        x, y = round(random.uniform(self.x_min, self.x_max), 1), round(random.uniform(self.y_min, self.y_max), 1) # random
        self.pos_sampling = [x, y]

        voronoi_polygon_img_output = os.path.join(self.fig_save_root, f"voronoi_step_{self.count}.png")
        graph_img_output = os.path.join(self.fig_save_root, f"graph_step_{self.count}.png")

        state = construct_graph_for_FOV(
            FOV_center=self.pos_sampling,
            range_radius=self.rs/2,
            cell_data=self.cell_data,
            voronoi_polygon_img_output=voronoi_polygon_img_output,
            graph_img_output=graph_img_output
        )

        print(self.pos_sampling)

        return state