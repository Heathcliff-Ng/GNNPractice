#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 20:12:57 2021

@author: Weiran
"""
import os
import time
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from scipy.spatial import Delaunay


NEIGHBOR_EDGE_CUTOFF = 55  # distance cutoff for neighbor edges, 55 pixels~20 um



def plot_voronoi_polygons(voronoi_polygons, voronoi_polygon_colors=None):
    """Plot voronoi polygons for the cellular graph

    Args:
        voronoi_polygons (nx.Graph/list): cellular graph or list of voronoi polygons
        voronoi_polygon_colors (list): list of colors for voronoi polygons
    """
    if isinstance(voronoi_polygons, nx.Graph):
        voronoi_polygons = [voronoi_polygons.nodes[n]['voronoi_polygon'] for n in voronoi_polygons.nodes]

    if voronoi_polygon_colors is None:
        voronoi_polygon_colors = ['w'] * len(voronoi_polygons)
    assert len(voronoi_polygon_colors) == len(voronoi_polygons)

    xmax = 0
    ymax = 0
    for polygon, polygon_color in zip(voronoi_polygons, voronoi_polygon_colors):
        x, y = polygon[:, 0], polygon[:, 1]
        plt.fill(x, y, facecolor=polygon_color, edgecolor='k', linewidth=0.5)
        xmax = max(xmax, x.max())
        ymax = max(ymax, y.max())

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    return


def plot_graph(G, node_colors=None):
    """Plot dot-line graph for the cellular graph

    Args:
        G (nx.Graph): full cellular graph of the region
        node_colors (list): list of node colors. Defaults to None.
    """
    # Extract basic node attributes
    node_coords = [G.nodes[n]['center_coord'] for n in G.nodes]
    node_coords = np.stack(node_coords, 0)

    if node_colors is None:
        unique_cell_types = sorted(set([G.nodes[n]['cell_type'] for n in G.nodes]))
        cell_type_to_color = {ct: matplotlib.cm.get_cmap("tab20")(i % 20) for i, ct in enumerate(unique_cell_types)}
        node_colors = [cell_type_to_color[G.nodes[n]['cell_type']] for n in G.nodes]
    assert len(node_colors) == node_coords.shape[0]

    for (i, j, edge_type) in G.edges.data():
        xi, yi = G.nodes[i]['center_coord']
        xj, yj = G.nodes[j]['center_coord']
        if edge_type['edge_type'] == 'neighbor':
            plotting_kwargs = {"c": "k",
                               "linewidth": 1,
                               "linestyle": '-'}
        else:
            plotting_kwargs = {"c": (0.4, 0.4, 0.4, 1.0),
                               "linewidth": 0.3,
                               "linestyle": '--'}
        plt.plot([xi, xj], [yi, yj], zorder=1, **plotting_kwargs)

    plt.scatter(node_coords[:, 0],
                node_coords[:, 1],
                s=10,
                c=node_colors,
                linewidths=0.3,
                zorder=2)
    plt.xlim(0, node_coords[:, 0].max() * 1.01)
    plt.ylim(0, node_coords[:, 1].max() * 1.01)
    return


def load_cell_coords(cell_coords_file):
    """Load cell coordinates from file

    Args:
        cell_coords_file (str): path to csv file containing cell coordinates

    Returns:
        pd.DataFrame: dataframe containing cell coordinates, columns ['CELL_ID', 'X', 'Y']
    """
    df = pd.read_csv(cell_coords_file)
    df.columns = [c.upper() for c in df.columns]
    assert 'X' in df.columns, "Cannot find column for X coordinates"
    assert 'Y' in df.columns, "Cannot find column for Y coordinates"
    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    return df[['CELL_ID', 'X', 'Y']]


def load_cell_types(cell_types_file):
    """Load cell types from file

    Args:
        cell_types_file (str): path to csv file containing cell types

    Returns:
        pd.DataFrame: dataframe containing cell types, columns ['CELL_ID', 'CELL_TYPE']
    """
    df = pd.read_csv(cell_types_file)
    df.columns = [c.upper() for c in df.columns]

    cell_type_column = [c for c in df.columns if c != 'CELL_ID']
    if len(cell_type_column) == 1:
        cell_type_column = cell_type_column[0]
    elif 'CELL_TYPE' in cell_type_column:
        cell_type_column = 'CELL_TYPE'
    elif 'CELL_TYPES' in cell_type_column:
        cell_type_column = 'CELL_TYPES'
    else:
        raise ValueError("Please rename the column for cell type as 'CELL_TYPE'")

    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    _df = df[['CELL_ID', cell_type_column]]
    _df.columns = ['CELL_ID', 'CELL_TYPE']  # rename columns for clarity
    return _df


def load_cell_biomarker_expression(cell_biomarker_expression_file):
    """Load cell biomarker expression from file

    Args:
        cell_biomarker_expression_file (str): path to csv file containing cell biomarker expression

    Returns:
        pd.DataFrame: dataframe containing cell biomarker expression,
            columns ['CELL_ID', 'BM-<biomarker1_name>', 'BM-<biomarker2_name>', ...]
    """
    df = pd.read_csv(cell_biomarker_expression_file)
    df.columns = [c.upper() for c in df.columns]
    biomarkers = sorted([c for c in df.columns if c != 'CELL_ID'])
    for bm in biomarkers:
        if df[bm].dtype not in [np.dtype(int), np.dtype(float), np.dtype('float64')]:
            warnings.warn("Skipping column %s as it is not numeric" % bm)
            biomarkers.remove(bm)

    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    _df = df[['CELL_ID'] + biomarkers]
    _df.columns = ['CELL_ID'] + ['BM-%s' % bm for bm in biomarkers]
    return _df


def load_cell_features(cell_features_file):
    """Load additional cell features from file

    Args:
        cell_features_file (str): path to csv file containing additional cell features

    Returns:
        pd.DataFrame: dataframe containing cell features
            columns ['CELL_ID', '<feature1_name>', '<feature2_name>', ...]
    """
    df = pd.read_csv(cell_features_file)
    df.columns = [c.upper() for c in df.columns]

    feature_columns = sorted([c for c in df.columns if c != 'CELL_ID'])
    for feat in feature_columns:
        if df[feat].dtype not in [np.dtype(int), np.dtype(float), np.dtype('float64')]:
            warnings.warn("Skipping column %s as it is not numeric" % feat)
            feature_columns.remove(feat)

    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index

    return df[['CELL_ID'] + feature_columns]


def calcualte_voronoi_from_coords(x, y, xmax=None, ymax=None):
    """Calculate voronoi polygons from a set of points

    Points are assumed to have coordinates in ([0, xmax], [0, ymax])

    Args:
        x (array-like): x coordinates of points
        y (array-like): y coordinates of points
        xmax (float): maximum x coordinate
        ymax (float): maximum y coordinate

    Returns:
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices
    """
    from geovoronoi import voronoi_regions_from_coords
    from shapely import geometry
    xmax = 1.01 * max(x) if xmax is None else xmax
    ymax = 1.01 * max(y) if ymax is None else ymax
    boundary = geometry.Polygon([[0, 0], [xmax, 0], [xmax, ymax], [0, ymax]])
    coords = np.stack([
        np.array(x).reshape((-1,)),
        np.array(y).reshape((-1,))], 1)
    region_polys, _ = voronoi_regions_from_coords(coords, boundary)
    voronoi_polygons = [np.array(list(region_polys[k].exterior.coords)) for k in region_polys]
    return voronoi_polygons


def build_graph_from_cell_coords(cell_data, voronoi_polygons):
    """Construct a networkx graph based on cell coordinates

    Args:
        cell_data (pd.DataFrame): dataframe containing cell data,
            columns ['CELL_ID', 'X', 'Y', ...]
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices

    Returns:
        G (nx.Graph): full cellular graph of the region
    """
    save_polygon = True
    if not len(cell_data) == len(voronoi_polygons):
        warnings.warn("Number of cells does not match number of voronoi polygons")
        save_polygon = False

    coord_ar = np.array(cell_data[['CELL_ID', 'X', 'Y']])
    G = nx.Graph()
    node_to_cell_mapping = {}
    for i, row in enumerate(coord_ar):
        vp = voronoi_polygons[i] if save_polygon else None
        G.add_node(i, voronoi_polygon=vp)
        node_to_cell_mapping[i] = row[0]

    dln = Delaunay(coord_ar[:, 1:3])
    neighbors = [set() for _ in range(len(coord_ar))]
    for t in dln.simplices:
        for v in t:
            neighbors[v].update(t)

    for i, ns in enumerate(neighbors):
        for n in ns:
            G.add_edge(int(i), int(n))

    return G, node_to_cell_mapping


def assign_attributes(G, cell_data, node_to_cell_mapping):
    """Assign node and edge attributes to the cellular graph

    Args:
        G (nx.Graph): full cellular graph of the region
        cell_data (pd.DataFrame): dataframe containing cellular data
        node_to_cell_mapping (dict): 1-to-1 mapping between
            node index in `G` and cell id

    Returns:
        nx.Graph: populated cellular graph
    """
    assert set(G.nodes) == set(node_to_cell_mapping.keys())
    biomarkers = sorted([c for c in cell_data.columns if c.startswith('BM-')])

    additional_features = sorted([
        c for c in cell_data.columns if c not in biomarkers + ['CELL_ID', 'X', 'Y', 'CELL_TYPE']])

    cell_to_node_mapping = {v: k for k, v in node_to_cell_mapping.items()}
    node_properties = {}
    for _, cell_row in cell_data.iterrows():
        cell_id = cell_row['CELL_ID']
        if cell_id not in cell_to_node_mapping:
            continue
        node_index = cell_to_node_mapping[cell_id]
        p = {"cell_id": cell_id}
        p["center_coord"] = (cell_row['X'], cell_row['Y'])
        if "CELL_TYPE" in cell_row:
            p["cell_type"] = cell_row["CELL_TYPE"]
        else:
            p["cell_type"] = "Unassigned"
        biomarker_expression_dict = {bm.split('BM-')[1]: cell_row[bm] for bm in biomarkers}
        p["biomarker_expression"] = biomarker_expression_dict
        for feat_name in additional_features:
            p[feat_name] = cell_row[feat_name]
        node_properties[node_index] = p

    nx.set_node_attributes(G, node_properties)

    # Add distance, edge type (by thresholding) to edge feature
    edge_properties = get_edge_type(G)
    nx.set_edge_attributes(G, edge_properties)
    return G


def get_edge_type(G, neighbor_edge_cutoff=NEIGHBOR_EDGE_CUTOFF):
    """Define neighbor vs distant edges based on distance

    Args:
        G (nx.Graph): full cellular graph of the region
        neighbor_edge_cutoff (float): distance cutoff for neighbor edges.
            By default we use 55 pixels (~20 um)

    Returns:
        dict: edge properties
    """
    edge_properties = {}
    for (i, j) in G.edges:
        ci = G.nodes[i]['center_coord']
        cj = G.nodes[j]['center_coord']
        dist = np.linalg.norm(np.array(ci) - np.array(cj), ord=2)
        edge_properties[(i, j)] = {
            "distance": dist,
            "edge_type": "neighbor" if dist < neighbor_edge_cutoff else "distant"
        }
    return edge_properties


def merge_cell_dataframes(df1, df2):
    """Merge two cell dataframes on shared rows (cells)"""
    if set(df2['CELL_ID']) != set(df1['CELL_ID']):
        warnings.warn("Cell ids in the two dataframes do not match")
    shared_cell_ids = set(df2['CELL_ID']).intersection(set(df1['CELL_ID']))
    df1 = df1[df1['CELL_ID'].isin(shared_cell_ids)]
    df1 = df1.merge(df2, on='CELL_ID')
    return df1

def load_cell_data(cell_coords_file=None, cell_types_file=None,
                   cell_biomarker_expression_file=None, cell_features_file=None):
    """Load cellular data for graph construct

    Args:
        cell_coords_file (str): path to csv file containing cell coordinates
        cell_types_file (str): path to csv file containing cell types/annotations
        cell_biomarker_expression_file (str): path to csv file containing cell biomarker expression
        cell_features_file (str): path to csv file containing additional cell features
            Note that features stored in this file can only be numeric and
            will be saved and used as is

    Returns:
        pd.DataFrame: dataframe containing cell features
    """
    assert cell_coords_file is not None, "cell coordinates must be provided"
    cell_data = load_cell_coords(cell_coords_file)


    if cell_types_file is not None:
        # Load cell types
        cell_types = load_cell_types(cell_types_file)
        cell_data = merge_cell_dataframes(cell_data, cell_types)

    if cell_biomarker_expression_file is not None:
        # Load cell biomarker expression
        cell_expression = load_cell_biomarker_expression(cell_biomarker_expression_file)
        cell_data = merge_cell_dataframes(cell_data, cell_expression)

    if cell_features_file is not None:
        # Load additional cell features
        additional_cell_features = load_cell_features(cell_features_file)
        cell_data = merge_cell_dataframes(cell_data, additional_cell_features)

    return cell_data


def construct_graph_for_FOV(FOV_center=None,
                            range_radius=100,
                            cell_data=None,
                            graph_output=None,
                            voronoi_polygon_img_output=None,
                            graph_img_output=None,
                            figsize=10):
    """Construct cellular graph for a specific FOV_center

    Args:
        FOV_center (list): coordinates [x_center, y_center] defining the center of the Field of View (FOV).
        range_radius (int): radius around the FOV center within which cells are included in the graph.
        cell_data (pd.DataFrame): DataFrame containing cell data with columns ['CELL_ID', 'X', 'Y', ...].
        graph_output (str): path for saving cellular graph as gpickle
        voronoi_polygon_img_output (str): path for saving voronoi image
        graph_img_output (str): path for saving dot-line graph image
        figsize (int): figure size for plotting

    Returns:
        G (nx.Graph): full cellular graph of the region
    """

    # record start time
    start_time = time.time()

    x_center, y_center = FOV_center

    # points in range
    cell_data = cell_data[
        (cell_data['X'] >= x_center - range_radius) &
        (cell_data['X'] <= x_center + range_radius) &
        (cell_data['Y'] >= y_center - range_radius) &
        (cell_data['Y'] <= y_center + range_radius)
    ]

    # empty FOV check
    if cell_data.empty:
        G = nx.Graph()
        # set cell_id to 0；no cell_id makes x_matrix[0,:] all 0
        G.add_node(0, cell_id=None)
        G.FOV_center = FOV_center
        print("No cells found in FOV; returning single-node graph.")
        return G


    voronoi_polygons = calcualte_voronoi_from_coords(cell_data['X'], cell_data['Y'])

    G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, voronoi_polygons)


    # Assign attributes to cellular graph
    G = assign_attributes(G, cell_data, node_to_cell_mapping)
    G.FOV_center = FOV_center

    # Visualization of cellular graph
    if voronoi_polygon_img_output is not None:
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_voronoi_polygons(G)
        plt.axis('scaled')
        plt.savefig(voronoi_polygon_img_output, dpi=300, bbox_inches='tight')
    if graph_img_output is not None:
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_graph(G)
        plt.axis('scaled')
        plt.savefig(graph_img_output, dpi=300, bbox_inches='tight')

    # Save graph to file
    if graph_output is not None:
        with open(graph_output, 'wb') as f:
            pickle.dump(G, f)

    # record end time
    end_time = time.time()
    print(f"Time taken to construct the graph: {end_time - start_time:.2f} seconds")

    return G
