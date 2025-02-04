import logging
import math
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

from tqdm import tqdm
from shapely import length
from shapely.plotting import plot_line, plot_polygon
from shapely.geometry import LineString
from tchou_tchou.utils import vector_angle, angle_between_2_vectors
from tchou_tchou.raster_shortest_path import get_height_series, resample_raster


log = logging.getLogger(__name__)


class PSO:
    def __init__(
        self,
        hpi,
        vpi,
        costFunc,
        obstacle=None,
        minimum_radius=None,
        minimum_tangent_length=None,
        minimum_slope_length=None,
        minimum_slope=None,
        maximum_slope=None,
        project=None,
        crs=4326,
    ):
        self.hpi = hpi
        self.vpi = vpi
        self.velh = pd.DataFrame(
            np.zeros(hpi.shape), columns=hpi.columns, index=hpi.index
        )
        self.velv = pd.DataFrame(
            np.random.uniform(-0.001, 0.001, vpi.shape),
            columns=vpi.columns,
            index=vpi.index,
        )
        self.costFunc = costFunc
        self.obstacle = obstacle
        self.costs = None
        self.pbest_hpi = hpi
        self.pbest_vpi = vpi
        self.pbest_cost = pd.Series([np.inf] * len(hpi.index), index=hpi.index)
        self.gbest_hpi = None
        self.gbest_vpi = None
        self.gbest_cost = np.inf
        self.iter = []
        self.linestrings = _alignment(self.hpi)
        self.minimum_radius = minimum_radius
        self.minimum_tangent_length = minimum_tangent_length
        self.minimum_slope_length = minimum_slope_length
        self.minimum_slope = minimum_slope
        self.maximum_slope = maximum_slope
        self.project = project
        self.crs = crs

        self.mean_vel = []
        self.artists = []

        self.velh[self.velh.columns.get_level_values(0)[0]] = 0.0
        self.velh[self.velh.columns.get_level_values(0)[-1]] = 0.0
        self.velv[self.velv.columns.get_level_values(0)[0]] = 0.0
        self.velv[self.velv.columns.get_level_values(0)[-1]] = 0.0

    def run(
        self,
        max_iter=1000,
        stalling=200,
        log_frequency=50,
        vmax=None,
    ):
        for t in tqdm(range(1, max_iter + 1)):
            self.costs = self.costFunc(self)

            # Update global best
            if min(self.costs) < self.gbest_cost:
                log.debug(f"New global best: {self.costs.idxmin()}")
                self.gbest_hpi = self.hpi.loc[self.costs.idxmin()].copy()
                self.gbest_vpi = self.vpi.loc[self.costs.idxmin()].copy()
                self.gbest_cost = min(self.costs)
            self.iter.append(self.gbest_cost)

            # Convergence
            if t % log_frequency == 0:
                log.info(f"Iteration {t}: {self.gbest_cost}")
            if t > stalling:
                if np.diff(self.iter[-stalling:]).sum() > -1:
                    log.info(
                        f"Stopping because it is stalled ({stalling} it w/o improvements)"
                    )
                    return 1

            # Update personal bests
            pbest_mask = self.costs < self.pbest_cost
            if sum(pbest_mask) > 0:
                log.debug(f"New personal bests for: {set(pbest_mask.index)}")
                self.pbest_hpi.loc[pbest_mask] = self.hpi.copy()
                self.pbest_vpi.loc[pbest_mask] = self.vpi.copy()
                self.pbest_cost.loc[pbest_mask] = self.costs.copy()

            # Update Velocity
            w = (
                -2.25 * t**3 / (7 * max_iter**3)
                + 6.75 * t**2 / (7 * max_iter**2)
                - 8 * t / (7 * max_iter)
                + 1
            )
            c1 = 2.0
            c2 = 2.0

            r1 = np.random.random(size=self.velh.shape)
            r2 = np.random.random(size=self.velh.shape)
            vel_cognitive = c1 * r1 * (self.pbest_hpi - self.hpi)
            vel_social = c2 * r2 * (self.gbest_hpi - self.hpi)
            self.velh = w * self.velh + vel_cognitive + vel_social

            r1 = np.random.random(size=self.velv.shape)
            r2 = np.random.random(size=self.velv.shape)
            vel_cognitive = c1 * r1 * (self.pbest_vpi - self.vpi)
            vel_social = c2 * r2 * (self.gbest_vpi - self.vpi)
            self.velv = w * self.velv + vel_cognitive + vel_social

            if (np.abs(self.velh).mean().sum() + np.abs(self.velv).mean().sum()) < 1:
                log.info("Solution converged (No more velocity)")
                return 0

            if vmax:
                self.velh = self.velh.clip(lower=-vmax, upper=vmax)

            VX = self.velh.swaplevel(axis=1)["X"]
            VY = self.velh.swaplevel(axis=1)["Y"]
            V = np.sqrt(VX**2 + VY**2).mean(axis=1)
            self.mean_vel.append(V.values)

            # Update Position
            hpi = self.hpi + self.velh
            vpi = self.vpi + self.velv
            maximum_radius = compute_maximum_radius(hpi, self.minimum_tangent_length)
            hpi = self._limit_radius(hpi, maximum_radius)
            vpi = self._limit_height(vpi)

            is_valid = self._hpi_is_valid(hpi)
            if sum(is_valid) >= 1:
                self.hpi.loc[is_valid] = hpi
                self.linestrings.loc[is_valid] = _alignment(self.hpi.loc[is_valid])

            if sum(~is_valid) >= 1:
                self.velh.loc[~is_valid] = 0

            is_valid = self._vpi_is_valid(vpi)
            if sum(is_valid) >= 1:
                self.vpi.loc[is_valid] = vpi

            if sum(~is_valid) >= 1:
                self.velv.loc[~is_valid] = 0

            # log.debug(f'Best particule is {self.gbest_part.name}')
            # self.linestrings.reset_index().to_file(f'alignments_{t}.geojson')
            self._plot(t)
            self._plot_vertical(t)

    def _hpi_is_valid(self, hpi):
        # Check radius constraint
        if self.minimum_radius:
            is_radius_valid = check_minimum_radius(hpi, self.minimum_radius)
            if sum(~is_radius_valid):
                log.debug(f"{sum(~is_radius_valid)} particles with invalid radius")
        else:
            is_radius_valid = pd.Series([True] * len(hpi.index), index=hpi.index)

        alignments = _alignment(hpi)

        # Check contact with obstacles
        if self.obstacle:
            is_touching = check_contact_with_obstacles(alignments, self.obstacle)
            if sum(is_touching):
                log.debug(f"{sum(is_touching)} particles are touching obstacle")
        else:
            is_touching = pd.Series([False] * len(hpi.index), index=hpi.index)

        # Check tangent length
        # if self.minimum_tangent_length:
        #     is_tangent_valid = pd.Series([all(tl >= self.minimum_tangent_length) for tl in alignments['tangents_length']], index=hpi.index)
        #     if sum(~is_tangent_valid):
        #         log.debug(f"{sum(~is_tangent_valid)} particles with invalid tangent")
        # else:
        #     is_tangent_valid = pd.Series([True] * len(hpi.index), index=hpi.index)

        return ~is_touching & is_radius_valid  # & is_tangent_valid

    def _vpi_is_valid(self, vpi):
        # Check slope
        if (self.minimum_slope is not None) | (self.maximum_slope is not None):
            is_valid_slope = check_slope(
                vpi, self.linestrings, self.minimum_slope, self.maximum_slope
            )
            if sum(~is_valid_slope):
                log.debug(f"{sum(~is_valid_slope)} particles with invalid slope")

        # Check minimum slope length
        if self.minimum_slope_length:
            is_valid_slope_length = check_slope_length(
                vpi, self.linestrings, self.minimum_slope_length
            )
            if sum(~is_valid_slope_length):
                log.debug(
                    f"{sum(~is_valid_slope_length)} particles with invalid slope length"
                )

        return is_valid_slope_length

    def _limit_radius(self, hpi, maximum_radius):
        hpi = hpi.swaplevel(axis=1)
        radius = hpi["R"]
        clipped = np.clip(radius, a_max=maximum_radius, a_min=None).fillna(0.0)
        hpi.loc[:, "R"] = clipped.values
        return hpi.swaplevel(axis=1)

    def _limit_height(self, vpi):
        stations = vpi.swaplevel(axis=1)["K"]
        stations = stations.mul(self.linestrings.length, axis=0)
        h = self.vpi.swaplevel(axis=1)["H"]
        ds = stations.diff(axis=1)

        dh_min_slope = self.minimum_slope * ds
        dh_max_slope = self.maximum_slope * ds

        df1 = (h - dh_max_slope).shift(-1, axis=1)
        df2 = h.shift(1, axis=1) + dh_min_slope
        dh_min = (df1.where(df1 > df2, df2).fillna(df1).fillna(df2) - h) / 2
        h_min = dh_min + h

        df1 = (h - dh_min_slope).shift(-1, axis=1)
        df2 = h.shift(1, axis=1) + dh_max_slope
        dh_max = (df1.where(df1 < df2, df2).fillna(df1).fillna(df2) - h) / 2
        h_max = dh_max + h

        vpi.loc[:, (slice(None), "H")] = vpi.loc[:, (slice(None), "H")].values.clip(
            h_min.values, h_max.values
        )

        return vpi

    def _plot(self, t):
        fig, ax = plt.subplots(figsize=(6, 10))
        for line in self.linestrings.geometry.values:
            plot_line(line, ax=ax, add_points=False, color="white")
        plot_line(
            self.linestrings.loc[self.gbest_hpi.name].geometry,
            ax=ax,
            add_points=False,
            color="red",
        )
        if self.obstacle:
            plot_polygon(self.obstacle, ax=ax, add_points=False, color="red")
        ctx.add_basemap(
            ax, crs=self.crs, source=f"../local/{self.project}/DEM_zone.tif"
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(f"../local/{self.project}/outputs/images/{t}.png")
        plt.close(fig)

    def _plot_vertical(self, t):
        fig, ax = plt.subplots(figsize=(20, 10))

        def redistribute_vertices(geom, distance):
            num_vert = int(round(geom.length / distance))
            return LineString(
                [
                    geom.interpolate(float(n) / num_vert, normalized=True)
                    for n in range(num_vert + 1)
                ]
            )

        rasters = {}
        with rasterio.open(f"../local/{self.project}/DEM_zone.tif") as src:
            for i in range(1, 20):
                with resample_raster(src, scale=1 / i) as raster:
                    raster.data = raster.read(1).astype(int)
                    raster.dataframe = pd.DataFrame(raster.data)
                    rasters[i] = raster

        bgeom = self.linestrings.loc[self.gbest_hpi.name].geometry
        gpath = redistribute_vertices(bgeom, distance=60)
        hs = get_height_series(gpath, raster=rasters[1])

        ax = hs.plot(figsize=(15, 5))
        profile = self.gbest_vpi.unstack().set_index("K")["H"]
        profile.index *= gpath.length
        profile.plot(color="red", ax=ax)
        plt.tight_layout()
        fig.savefig(f"../local/{self.project}/outputs/images/V_{t}.png")
        plt.close(fig)


def _linestring(hpi):
    df = hpi.T.unstack(level=1).swaplevel(axis=1)
    ls = [
        LineString([(a, b) for a, b in zip(row1, row2)])
        for row1, row2 in zip(df["X"].T.values, df["Y"].T.values)
    ]
    return ls


def _alignment(hpi):
    ls_list = _linestring(hpi)
    alignments = []
    tangents_index = []
    tangents_length = []
    for ls, R in zip(ls_list, hpi.swaplevel(axis=1)["R"].iterrows()):
        R = R[1].to_list()
        coords = [ls.coords[0]]
        indexes = [0]
        for i in range(1, len(ls.coords) - 1):
            P0 = ls.coords[i - 1]
            P1 = ls.coords[i]
            P2 = ls.coords[i + 1]
            if (
                abs(
                    angle_between_2_vectors(np.subtract(P0, P1), np.subtract(P2, P1))
                    - np.pi
                )
                < 0.1
            ):
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
    return gpd.GeoDataFrame(
        {
            "tangents_index": tangents_index,
            "tangents_length": tangents_length,
            "geometry": alignments,
        },
        index=hpi.index,
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

        increments = 0.1
        theta = []
        while len(theta) <= 6:
            if abs(angle1 - angle2) > np.pi:
                if angle1 >= np.pi:
                    angle1 -= 2 * np.pi
                else:
                    angle2 -= 2 * np.pi
            theta = np.linspace(
                angle1, angle2, abs(int((angle2 - angle1) / increments))
            )
            increments /= 2
        x = c[0] + r * np.cos(theta)
        y = c[1] + r * np.sin(theta)
    except ValueError:
        print(P0)
        print(P1)
        print(P2)
        print(r)
    return LineString(np.column_stack([x, y]))


def compute_maximum_radius(hpi, minimum_length):
    linestrings = _linestring(hpi)
    segments_df = pd.DataFrame(
        [
            list(map(LineString, zip(ls.coords[:-1], ls.coords[1:])))
            for ls in linestrings
        ]
    ).T
    length_df = segments_df.map(length)

    def vectorize(x):
        return np.subtract(x.coords[1], x.coords[0])

    vector_df = segments_df.map(vectorize)
    incoming_vector = vector_df.iloc[0:-1].values
    outgoing_vector = vector_df.shift(-1).iloc[0:-1].values

    alpha_df = pd.DataFrame(
        np.pi - np.vectorize(angle_between_2_vectors)(incoming_vector, outgoing_vector),
        columns=vector_df.columns,
    )

    is_aligned = abs(alpha_df - np.pi) < 0.1
    length_df = (
        length_df
        + length_df[is_aligned.shift()].shift(-1).fillna(0.0)
        + length_df[is_aligned].shift().fillna(0.0)
    )
    incoming_length = length_df.iloc[0:-1].values
    outgoing_length = length_df.shift(-1).iloc[0:-1].values
    min_length_df = pd.DataFrame(
        np.minimum(incoming_length, outgoing_length), columns=length_df.columns
    )

    maximum_radius = ((min_length_df - minimum_length) / 2) * np.tan(alpha_df / 2)
    maximum_radius.index += 1
    maximum_radius = maximum_radius.T
    maximum_radius.index = hpi.index

    return maximum_radius


def check_minimum_radius(hpi, minimum_radius):
    radius = hpi.swaplevel(axis=1)["R"]
    return (radius[radius.columns[1:-1]] >= minimum_radius).all(axis=1)


def check_contact_with_obstacles(alignments, obstacle):
    return pd.Series(
        [obstacle.intersects(ls) for ls in alignments.geometry], index=alignments.index
    )


def check_slope_length(vpi, alignments, minimum_slope_length):
    lengths = alignments.length
    stations = vpi.swaplevel(axis=1)["K"]
    slope_length = stations.mul(lengths, axis=0).diff(axis=1).drop(columns=0)
    return (slope_length >= minimum_slope_length).all(axis=1)


def check_slope(vpi, alignments, minimum_slope, maximum_slope):
    stations = vpi.swaplevel(axis=1)["K"]
    stations = stations.mul(alignments.length, axis=0)
    height = vpi.swaplevel(axis=1)["H"]
    slope = (height.diff(axis=1) / stations.diff(axis=1)).drop(columns=0)
    slope = np.round(slope, -6)
    if not minimum_slope:
        minimum_slope = -np.inf
    if not maximum_slope:
        maximum_slope = np.inf
    return ((slope >= minimum_slope) & (slope <= maximum_slope)).all(axis=1)
