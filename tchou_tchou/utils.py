import numpy as np
from shapely.geometry import LineString


def vector_angle(V):
    '''
    Compute the absolute angle of a vector between 0 and 2 * pi
    '''
    angle = np.arctan2(V[1], V[0])
    if angle < 0:
        return 2 * np.pi + angle
    else:
        return angle


def angle_between_2_vectors(V1, V2, return_unit_vector=False):
    if (np.sqrt(V1[0] ** 2 + V1[1] ** 2) - 1) > 1e-3:
        V1 /= np.sqrt(V1[0] ** 2 + V1[1] ** 2)
    if (np.sqrt(V2[0] ** 2 + V2[1] ** 2) - 1) > 1e-3:
        V2 /= np.sqrt(V2[0] ** 2 + V2[1] ** 2)

    a = np.arccos(np.clip(np.dot(V1, V2), -1.0, 1.0))
    if return_unit_vector:
        return a, V1, V2
    else:
        return a

def redistribute_vertices(geom, distance):
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))
