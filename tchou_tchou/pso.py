import logging
import random
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from shapely.geometry import LineString
from tchou_tchou.utils import vector_angle


log = logging.getLogger(__name__)


class PSO():
    def __init__(self,
                 pos,
                 costFunc,
                 obstacle=None,
                 minimum_radius=None,
                 max_iter=1000):

        self.pos = pos
        self.vel = pd.DataFrame(np.random.uniform(-100, 100, size=pos.shape), columns=pos.columns, index=pos.index)
        self.costFunc = costFunc
        self.obstacle = obstacle
        self.max_iter = max_iter
        self.costs = None
        self.pbest_part = pos
        self.pbest_cost = pd.Series([np.inf] * len(pos.index), index=pos.index)
        self.gbest_part = None
        self.gbest_cost = np.inf
        self.iter = []
        self.linestrings = _alignment(self.pos)
        self.minimum_radius = minimum_radius

        self.vel[self.vel.columns.get_level_values(0)[0]] = 0
        self.vel[self.vel.columns.get_level_values(0)[-1]] = 0

    def run(self):
        for t in tqdm(range(self.max_iter)):
            self.costs = self.costFunc(self)

            # Update global best
            if min(self.costs) < self.gbest_cost:
                self.gbest_part = self.pos.loc[self.costs.idxmin()]
                self.gbest_cost = min(self.costs)
            self.iter.append(self.gbest_cost)

            # Update personal bests
            self.pbest_part.loc[self.costs < self.pbest_cost] = self.pos
            self.pbest_cost.loc[self.costs < self.pbest_cost] = self.costs

            if t % 10 == 0:
                log.info(f"Iteration {t}: {self.gbest_cost}")

            # Update Velocity
            w = - 2.25 * t ** 3 / (7 * self.max_iter ** 3) \
                + 6.75 * t ** 2 / (7 * self.max_iter ** 2) \
                - 8 * t / (7 * self.max_iter) + 1
            c1 = 2
            c2 = 2
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pbest_part - self.pos)
            vel_social = c2 * r2 * (self.gbest_part - self.pos)
            self.vel = w * self.vel + vel_cognitive + vel_social

            # Update Position
            pos = self.pos + self.vel
            is_valid = self._is_valid(pos)
            self.pos[is_valid] = pos

    def _is_valid(self, pos):
        # Check radius constraint
        if self.minimum_radius:
            radius = pos.swaplevel(axis=1)['R']
            is_radius_valid = (radius[radius.columns[1:-1]] >= self.minimum_radius).all(axis=1)
        else:
            is_radius_valid = pd.Series([True] * len(pos.index), index=pos.index)

        # Check contact with obstacles
        if self.obstacle:
            is_touching = pd.Series([self.obstacle.intersects(ls) for ls in _alignment(pos)], index=pos.index)
        else:
            is_touching = pd.Series([False] * len(pos.index), index=pos.index)

        return ~is_touching & is_radius_valid


def _linestring(pos):
    df = pos.T.unstack(level=1).swaplevel(axis=1)
    ls = [LineString([(a, b) for a, b in zip(row1, row2)]) for row1, row2 in zip(df['X'].T.values, df['Y'].T.values)]
    return ls


def _alignment(pos):
    ls_list = _linestring(pos)
    alignments = []
    for ls, R in zip(ls_list, pos.swaplevel(axis=1)['R'].iterrows()):
        R = R[1].to_list()
        coords = [ls.coords[0]]
        for i in range(1, len(ls.coords) - 1):
            arc = arc_between_3pts(ls.coords[i - 1], ls.coords[i], ls.coords[i + 1], R[i])
            coords += arc.coords
        coords += [ls.coords[-1]]
        alignments.append(LineString(coords))
    return gpd.GeoSeries(alignments, index=pos.index)


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
            if (angle1 - angle2) > np.pi:
                angle1 -= 2 * np.pi
            theta = np.linspace(angle1, angle2, abs(int((angle2 - angle1) / increments)))
            increments /= 2
        x = c[0] + r * np.cos(theta)
        y = c[1] + r * np.sin(theta)

        x = c[0] + r * np.cos(theta)
        y = c[1] + r * np.sin(theta)

    except ValueError:
        print(P0)
        print(P1)
        print(P2)
        print(r)
    return LineString(np.column_stack([x, y]))