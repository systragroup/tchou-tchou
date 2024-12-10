import os
import logging
import random
import shutil
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from tqdm import tqdm
from shapely import length
from shapely.plotting import plot_line, plot_polygon
from shapely.geometry import LineString
from tchou_tchou.utils import vector_angle, angle_between_2_vectors


log = logging.getLogger(__name__)


class PSO():
    def __init__(self,
                 pos,
                 costFunc,
                 obstacle=None,
                 minimum_radius=None,
                 minimum_tangent_length=None,):

        self.pos = pos
        self.vel = pd.DataFrame(np.zeros(pos.shape), columns=pos.columns, index=pos.index)
        self.costFunc = costFunc
        self.obstacle = obstacle
        self.costs = None
        self.pbest_part = pos
        self.pbest_cost = pd.Series([np.inf] * len(pos.index), index=pos.index)
        self.gbest_part = None
        self.gbest_cost = np.inf
        self.iter = []
        self.linestrings = _alignment(self.pos)
        self.minimum_radius = minimum_radius
        self.minimum_tangent_length = minimum_tangent_length

        #self.linestrings.to_file('alignments_0.geojson')

        self.mean_vel = []
        self.artists = []

        self.vel[self.vel.columns.get_level_values(0)[0]] = 0.0
        self.vel[self.vel.columns.get_level_values(0)[-1]] = 0.0

    def run(self,
            max_iter=1000,
            stalling=200,
            log_frequency=50,
            vmax=None,
            ):

        for t in tqdm(range(1, max_iter + 1)):
            self.costs = self.costFunc(self)

            # Update global best
            if min(self.costs) < self.gbest_cost:
                self.gbest_part = self.pos.loc[self.costs.idxmin()]
                self.gbest_cost = min(self.costs)
            self.iter.append(self.gbest_cost)

            # Convergence
            if t % log_frequency == 0:
                log.info(f"Iteration {t}: {self.gbest_cost}")
            if t > stalling:
                if np.diff(self.iter[-stalling:]).sum() > -1:
                    log.info(f"Stopping because it is stalled ({stalling} it w/o improvements)")
                    return 1

            # Update personal bests
            self.pbest_part.loc[self.costs < self.pbest_cost] = self.pos
            self.pbest_cost.loc[self.costs < self.pbest_cost] = self.costs

            # Update Velocity
            w = - 2.25 * t ** 3 / (7 * max_iter ** 3) \
                + 6.75 * t ** 2 / (7 * max_iter ** 2) \
                - 8 * t / (7 * max_iter) + 1
            c1 = 2.0
            c2 = 2.0
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pbest_part - self.pos)
            vel_social = c2 * r2 * (self.gbest_part - self.pos)
            self.vel = w * self.vel + vel_cognitive + vel_social

            if np.abs(self.vel).mean().sum() < 1:
                log.info("Solution converged (No more velocity)")
                return 0

            if vmax:
                self.vel = self.vel.clip(lower=-vmax, upper=vmax)

            VX = self.vel.swaplevel(axis=1)['X']
            VY = self.vel.swaplevel(axis=1)['Y']
            V = np.sqrt(VX ** 2 + VY ** 2).mean(axis=1)
            self.mean_vel.append(V.values)

            # Update Position
            pos = self.pos + self.vel
            maximum_radius = compute_maximum_radius(pos, self.minimum_tangent_length)
            pos = self._limit_radius(pos, maximum_radius)

            is_valid = self._is_valid(pos)
            if sum(is_valid) >= 1:
                self.pos.loc[is_valid] = pos
                self.linestrings.loc[is_valid] = _alignment(self.pos.loc[is_valid])

            if sum(~is_valid) >= 1:
                self.vel.loc[~is_valid] = 0

            #log.debug(f'Best particule is {self.gbest_part.name}')
            #self.linestrings.reset_index().to_file(f'alignments_{t}.geojson')
            self._plot(t)

    def _is_valid(self, pos):
        # Check radius constraint
        if self.minimum_radius:
            is_radius_valid = check_minimum_radius(pos, self.minimum_radius)
            if sum(~is_radius_valid):
                log.debug(f"{sum(~is_radius_valid)} particles with invalid radius")
        else:
            is_radius_valid = pd.Series([True] * len(pos.index), index=pos.index)

        alignments = _alignment(pos)

        # Check contact with obstacles
        if self.obstacle:
            is_touching = check_contact_with_obstacles(alignments, self.obstacle)
            if sum(is_touching):
                log.debug(f"{sum(is_touching)} particles are touching obstacle")
        else:
            is_touching = pd.Series([False] * len(pos.index), index=pos.index)

        # Check tangent length
        # if self.minimum_tangent_length:
        #     is_tangent_valid = pd.Series([all(tl >= self.minimum_tangent_length) for tl in alignments['tangents_length']], index=pos.index)
        #     if sum(~is_tangent_valid):
        #         log.debug(f"{sum(~is_tangent_valid)} particles with invalid tangent")
        # else:
        #     is_tangent_valid = pd.Series([True] * len(pos.index), index=pos.index)

        return ~is_touching & is_radius_valid  # & is_tangent_valid

    def _limit_radius(self, pos, maximum_radius):
        pos = pos.swaplevel(axis=1)
        radius = pos['R']
        clipped = np.clip(radius, a_max=maximum_radius, a_min=None).fillna(0.0)
        pos.loc[:, 'R'] = clipped.values
        return pos.swaplevel(axis=1)
    
    def _plot(self, t):
        fig, ax = plt.subplots(figsize=(6, 10))
        for line in self.linestrings.geometry.values:
            plot_line(line, ax=ax, add_points=False, color='white')
        plot_line(self.linestrings.loc[self.gbest_part.name].geometry, ax=ax, add_points=False, color='red')
        if self.obstacle:
            plot_polygon(self.obstacle, ax=ax, add_points=False, color='red')
        ctx.add_basemap(ax, crs=32628, source='../local/guinea/DEM_zone.tif')
        ax.set_aspect('equal')
        ax.set_axis_off()
        plt.tight_layout()
        self.artists.append(ax.images)
        fig.savefig(f'../images/{t}.png')
        plt.close(fig)


def _linestring(pos):
    df = pos.T.unstack(level=1).swaplevel(axis=1)
    ls = [LineString([(a, b) for a, b in zip(row1, row2)]) for row1, row2 in zip(df['X'].T.values, df['Y'].T.values)]
    return ls


def _alignment(pos):
    ls_list = _linestring(pos)
    alignments = []
    tangents_index = []
    tangents_length = []
    for ls, R in zip(ls_list, pos.swaplevel(axis=1)['R'].iterrows()):
        R = R[1].to_list()
        coords = [ls.coords[0]]
        indexes = [0]
        for i in range(1, len(ls.coords) - 1):
            P0 = ls.coords[i - 1]
            P1 = ls.coords[i]
            P2 = ls.coords[i + 1]
            if abs(angle_between_2_vectors(np.subtract(P0, P1), np.subtract(P2, P1)) - np.pi) < 0.1:
                indexes += [indexes[-1]]
            else:
                arc = arc_between_3pts(P0, P1, P2, R[i])
                coords += arc.coords
                indexes += [indexes[-1] + len(arc.coords)]
        coords += [ls.coords[-1]]
        lengths = [math.dist(coords[i + 1], coords[i]) for i in indexes]
        alignments.append(LineString(coords))
        tangents_index.append(indexes)
        tangents_length.append(np.round(lengths, 1))
    return gpd.GeoDataFrame({'tangents_index': tangents_index,
                             'tangents_length': tangents_length,
                             'geometry': alignments},
                            index=pos.index
                            )


def arc_between_3pts(P0, P1, P2, r):
    try:
        V1 = np.subtract(P0, P1)
        V2 = np.subtract(P2, P1)
        U1 = V1 / np.sqrt(V1[0] ** 2 + V1[1] ** 2)
        U2 = V2 / np.sqrt(V2[0] ** 2 + V2[1] ** 2)
        B = np.add(U1, U2)
        B /= np.sqrt(B[0] ** 2 + B[1] ** 2)
        alpha = np.arccos(np.clip(np.dot(U1, U2), -1.0, 1.0)) / 2
        d = r / np.sin(alpha)
        c = np.add(P1, B * d)
        c2 = np.add(P1, U2 * np.cos(alpha) * d)
        c1 = np.add(P1, U1 * np.cos(alpha) * d)
        CC1 = np.subtract(c1, c)
        CC2 = np.subtract(c2, c)
        angle1 = vector_angle(CC1)
        angle2 = vector_angle(CC2)

        increments = 0.02
        theta = []
        while len(theta) <= 6:
            if abs(angle1 - angle2) > np.pi:
                if angle1 >= np.pi:
                    angle1 -= 2 * np.pi
                else:
                    angle2 -= 2 * np.pi
            theta = np.linspace(angle1, angle2, abs(int((angle2 - angle1) / increments)))
            increments /= 2
        x = c[0] + r * np.cos(theta)
        y = c[1] + r * np.sin(theta)
    except ValueError:
        print(P0)
        print(P1)
        print(P2)
        print(r)
    return LineString(np.column_stack([x, y]))


def compute_maximum_radius(pos, minimum_length):
    linestrings = _linestring(pos)
    segments_df = pd.DataFrame([list(map(LineString, zip(ls.coords[:-1], ls.coords[1:]))) for ls in linestrings]).T
    length_df = segments_df.map(length)

    def vectorize(x):
        return np.subtract(x.coords[1], x.coords[0])

    vector_df = segments_df.map(vectorize)
    incoming_vector = vector_df.iloc[0:-1].values
    outgoing_vector = vector_df.shift(-1).iloc[0:-1].values

    alpha_df = pd.DataFrame(
        np.pi - np.vectorize(angle_between_2_vectors)(incoming_vector, outgoing_vector),
        columns=vector_df.columns
    )

    is_aligned = abs(alpha_df - np.pi) < 0.1
    length_df = length_df + length_df[is_aligned.shift()].shift(-1).fillna(0.0) + length_df[is_aligned].shift().fillna(0.0)
    incoming_length = length_df.iloc[0:-1].values
    outgoing_length = length_df.shift(-1).iloc[0:-1].values
    min_length_df = pd.DataFrame(np.minimum(incoming_length, outgoing_length), columns=length_df.columns)

    maximum_radius = ((min_length_df - minimum_length) / 2) * np.tan(alpha_df / 2)
    maximum_radius.index += 1
    maximum_radius = maximum_radius.T
    maximum_radius.index = pos.index

    return maximum_radius


def check_minimum_radius(pos, minimum_radius):
    radius = pos.swaplevel(axis=1)['R']
    return (radius[radius.columns[1:-1]] >= minimum_radius).all(axis=1)


def check_contact_with_obstacles(alignments, obstacle):
    return pd.Series([obstacle.intersects(ls) for ls in alignments.geometry], index=alignments.index)
