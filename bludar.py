
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
import requests
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def dowload_parquet(URL, filename=False):
    file_bytes = requests.get(URL).content

    #generic filename, just a number
    if (not filename):
        filename = str(len(os.listdir())+1) + ".parquet"
    
    save_path = os.path.join(os.getcwd(), filename)
    
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    return True

def open_parquet(file_name = 'easy.parquet', save_results = True):
    points = pd.read_parquet(file_name)
    results = pd.DataFrame(do_all(points))
    if save_results:
        results.to_parquet('results_' + file_name)
    return results

def do_all(points):
    clusters = split_wires_by_peaks(points)

    catenary_params = []
    
    for cluster in clusters:
        catenary_params.append(fit_catenary_to_cluster(cluster))
    print('Process complete. Found: ' + str(len(clusters)))
    
    return {"detected_wires": len(clusters),
            "catenary_params": catenary_params}

def split_wires_by_peaks(points, bins=200, prominence=0.01, sigma=4):
    xy = points[["x", "y"]].to_numpy()
    pca = PCA(n_components=2)
    xy_rot = pca.fit_transform(xy)
    y_sep = xy_rot[:, 1]

    counts, edges = np.histogram(y_sep, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    smooth_counts = gaussian_filter1d(counts, sigma=sigma)

    abs_prom = prominence * np.max(smooth_counts)
    peak_indices, props = find_peaks(smooth_counts, prominence=abs_prom)

    peak_positions = centers[peak_indices]

    distances = np.abs(y_sep[:, None] - peak_positions[None, :])
    labels = np.argmin(distances, axis=1)

    wire_groups = []
    for i in range(len(peak_positions)):
        group = points[labels == i]
        wire_groups.append(group)

    return wire_groups

def fit_catenary_to_cluster(cluster):
    cluster_np = cluster[["x", "y", "z"]].to_numpy()

    # re-calculate best fit of direction for each cluster (redundant as they are all parallel)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(cluster_np[:, :2])
    direction = pca.components_[0]  # main axis in XY
    direction = direction / np.linalg.norm(direction)

    # dot product
    xy = cluster_np[:, :2]
    s = xy @ direction 

    [c, x0, y0], _ = curve_fit(
        catenary,
        s,
        cluster_np[:, 2],
        p0=[10, np.mean(s), np.mean(cluster_np[:, 2])]
    )

    x = s.max() - x0
    y = catenary(s.max(), c, x0, y0)
    
    # for notebook demo
    points_2D = np.column_stack([s, cluster_np[:, 2]])

    return_dict = {
        "c": c,
        "x": x,
        "y": y,
        "x0": x0,
        "y0": y0,
        "points": cluster_np.tolist(),
        "points_2D": points_2D.tolist()
    }
    
    return return_dict

def catenary(x, c, x0, y0):
    return y0 + c * (np.cosh((x - x0) / c) - 1)
