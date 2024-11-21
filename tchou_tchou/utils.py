import numpy as np


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

    a = np.arccos(np.clip(np.dot(V1, V2), -1.0, 1.0)) / 2
    if return_unit_vector:
        return a, V1, V2
    else:
        return a
    