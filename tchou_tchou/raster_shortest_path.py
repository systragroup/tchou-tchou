import rasterio
from rasterio import features
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely import geometry
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from rasterio.enums import Resampling
from contextlib import contextmanager  

import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling

# use context manager so DatasetReader and MemoryFile get cleaned up automatically
@contextmanager
def resample_raster(raster, scale=1/3):
    t = raster.transform
    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = int(raster.height * scale)
    width = int(raster.width * scale)

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = raster.read( # Note changed order of indexes, arrays are band, row, col order not row, col, band
            out_shape=(raster.count, height, width),
            resampling=Resampling.bilinear,
        )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset: # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return     

def get_delta_xy(data, dx=0, dy=0):
    data_t = data.copy()
    data_t.columns = [c - dx for c in data.columns]
    data_t.index = [i - dy for i in data.index]
    return (data_t-data).reindex(index=data.index, columns=data.columns)

def get_slopes(df):
    a = get_delta_xy(df, 0, -1)
    b = get_delta_xy(df, 0, 1)
    c = get_delta_xy(df, -1, 0)
    d = get_delta_xy(df, 1, 0)
    return pd.concat([np.abs(dz) for dz in [a, b, c, d]]).groupby(level=0).max()

def smooth_xy(df, **kwargs):
    return df.ewm(axis=1,  **kwargs).mean().ewm(axis=0, **kwargs).mean()

def get_impedance_raster(terrain, constraints, raster, inf=np.inf, pixel=10):
    layers = []
    for name, gdf in terrain.items():
        for distance, penalty in constraints[name].items():
            shapes = list(gdf['geometry'].buffer(distance+pixel))
            r = features.rasterize(shapes, fill=0, out_shape=raster.shape, transform=raster.transform)
            layers.append(penalty*r)
    impedance_raster = np.clip(sum([l.astype(int ) for l in layers]), 0, inf)
    return impedance_raster

def get_graph(
        impedance_dataframe, dem, 
        inf, scale=1, 
        asf=1, # anisotropic slope factor
        isf=1, # isotropic slope factor
        asmooth=0,
        ismooth=0,
        max_slope=10, # meters per cell
        scope = 2,
        smin=-0.7,
        smax=2,

    ):
    
    deltas = {}
    delta_range = list(range(-scope, scope+1))
    keep = impedance_dataframe.stack().loc[impedance_dataframe.stack()<inf].index

    # ANISOTROPIC
    if asf>0:
        if asmooth:
            data = smooth_xy(dem, halflife=asmooth)
        else:
            data = dem
        
        for dx in delta_range:
            for dy in delta_range:
                if dx != 0 or dy != 0 :
                    distance = np.sqrt(dx*dx + dy*dy)
                    dxdy_max_slope = max_slope*distance
                    delta = get_delta_xy(data, dx=dx, dy=dy)
                    kstack = delta.stack().loc[keep]
                    kstack = kstack.loc[kstack >= smin*distance]
                    kstack = kstack.loc[kstack <= smax*distance]
                    abs_stack = np.abs(kstack) # np.abs(delta).stack().loc[keep]
                    #abs_stack = abs_stack.loc[abs_stack<=dxdy_max_slope]
                    deltas[(dx, dy)] = abs_stack.to_dict()


    else: # asf == 0
        for dx in delta_range:
            for dy in delta_range:
                if dx != 0 or dy != 0 :
                    deltas[(dx, dy)] = {k : 0 for k in keep}

    imp = impedance_dataframe

    # ISOTROPIC
    if isf > 0:
        if ismooth:
            dem = smooth_xy(dem, halflife=ismooth)
        imp += isf*get_slopes(dem)

    imp = imp.clip(0, inf)
    stack = imp.stack()
    stack = stack.loc[stack<inf]

    edges = []

    for dxdy, delta in deltas.items():
        dx, dy = dxdy
        distance = np.sqrt(dx*dx + dy*dy)
        for ij, w in stack.to_dict().items():
            try:
                edges.append([ij, (ij[0]+dy, ij[1]+dx), w + scale*distance+ asf*delta[ij]])
            except KeyError : # the their is no delta for this ij+dx+dy
                pass

    g = nx.DiGraph()
    g.add_weighted_edges_from(edges)
    return g, imp, deltas

def get_path(
        terrain, constraints, raster,# Impedance raster
        checkpoints, checkpoints_buffer, strict_checkpoints=False, # search space
        scale=1, asf=1, isf=1, inf=np.inf, asmooth=0, ismooth=0, # slope raster and slope constraints
        scope=1, max_slope=np.inf, smax=np.inf, smin=-np.inf # slope in percent
        ):

    max_slope_meters = np.ceil(max_slope / 100 * raster.transform.a)
    smax_meters = np.ceil(smax / 100 * raster.transform.a)
    smin_meters = np.floor(smin / 100 * raster.transform.a)

    elevation = raster.dataframe
    # constraints and search space
    constraint_raster = get_impedance_raster(terrain=terrain, constraints=constraints, inf=inf, raster=raster)
    constraint_df = pd.DataFrame(constraint_raster)

    search_space = features.rasterize([checkpoints.buffer(checkpoints_buffer)], fill=inf, out_shape=raster.shape, transform=raster.transform)
    search_df = pd.DataFrame(search_space)
    combined = constraint_df + search_df 
    combined = combined.clip(0, inf)

    # add slopes and compute path
    g, imp, deltas = get_graph(
        impedance_dataframe=combined, dem=elevation, inf=inf, scale=scale, 
        asf=asf, isf=isf, asmooth=asmooth, ismooth=ismooth, scope=scope, max_slope=max_slope_meters, 
        smax=smax_meters, smin=smin_meters
    )
    c = list(checkpoints.coords)
    a, b = ab= [c[0], c[-1]] 
    if strict_checkpoints:
        ab = []
        for i in range(len(c)-1):
            ab.append([c[i], c[i+1]])
    else:
        ab= [[c[0], c[-1]]]
    full_path = []
    for a, b in ab:
        _, path = nx.bidirectional_dijkstra(g,raster.index(a[0], a[1]), raster.index(b[0], b[1]), weight='weight')
        full_path = full_path + path
    return imp, full_path, g, deltas

def get_height_series(path, raster):
    data = raster.data
    lengths = [path.project(geometry.Point(c)) for c in path.coords]
    heights = ([data[raster.index(x, y)] for x, y in path.coords])
    s = pd.Series(index=lengths, data=heights)
    s.name = 'height'
    s.index.name = 'length'
    return s

def read_parameters(file):
    parameter_frame = pd.read_excel(file, sheet_name='parameters').set_index(['category', 'parameter']).drop(['description', 'unit'], axis=1)
    types = parameter_frame['type'].dropna().to_dict()
    parameter_frame.drop(['type'], axis=1, inplace=True)
    var = parameter_frame

    def string_to_dict(constraint_string):
        return {
            int(s.split(':')[0]) : int(s.split(':')[1]) 
            for s in  constraint_string.strip('{ }').split(',')
        }
    for k, v in types.items():
        if v == 'float':
            var.loc[k] = var.loc[k].astype(float)
        elif v == 'int':
            var.loc[k] = var.loc[k].apply(int)
        elif v == 'bool':
            var.loc[k] = var.loc[k].apply(bool)
        elif v == 'str':
            var.loc[k] = var.loc[k].apply(str)
        elif v == 'json':
            var.loc[k] = var.loc[k].apply(lambda s: string_to_dict(s))
    return var