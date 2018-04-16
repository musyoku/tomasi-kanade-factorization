import os
import sys
import math
import random
import plotly.plotly as py
import plotly.graph_objs as graph
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tomasi_kanade


def main():
    object_points = np.asarray(
        [
            [
                0.0,
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
            ],
            [
                0.0,
                0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
            ],
            [
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
            ],
        ],
        dtype=np.float32)

    print(object_points)
    measurement_matrix_list_x = [[] for i in range(9)]
    measurement_matrix_list_y = [[] for i in range(9)]

    # ランダムな角度で正射影する
    for frame in range(100):
        random_rad_x = math.pi * random.uniform(-2.0, 2.0)
        random_rad_y = math.pi * random.uniform(-2.0, 2.0)
        rotation_matrix_axis_x = np.array(
            [[1.0, 0.0, 0.0], [
                0.0, math.cos(random_rad_x), -math.sin(random_rad_x)
            ], [0.0, math.sin(random_rad_x),
                math.cos(random_rad_x)]],
            dtype=np.float32)
        rotation_matrix_axis_y = np.array(
            [[math.cos(random_rad_y), 0.0, -math.sin(random_rad_y)], [
                0.0, 1.0, 0.0
            ], [math.sin(random_rad_y), 0.0,
                math.cos(random_rad_y)]],
            dtype=np.float32)
        rotation_matrix = np.dot(rotation_matrix_axis_x,
                                 rotation_matrix_axis_y)
        rotated_cube = np.dot(rotation_matrix, object_points)

        # 正射影
        for i in range(9):
            measurement_matrix_list_x[i].append(rotated_cube[0, i])
            measurement_matrix_list_y[i].append(rotated_cube[1, i])

    measurement_matrix_x = np.asarray(
        measurement_matrix_list_x, dtype=np.float32)
    measurement_matrix_y = np.asarray(
        measurement_matrix_list_y, dtype=np.float32)
    measurement_matrix = np.concatenate(
        (measurement_matrix_x.T, measurement_matrix_y.T), axis=0)

    registered_measurement_matrix = measurement_matrix - np.mean(
        measurement_matrix, axis=0)[None, :]

    print(registered_measurement_matrix, registered_measurement_matrix.shape)

    R, S, R_, S_ = tomasi_kanade.recover_3d_structure(registered_measurement_matrix)
    print(R.shape)
    print(S.shape)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_trisurf(S[0], S[1], S[2], linewidth=0.2, antialiased=True)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_trisurf(S_[0], S_[1], S_[2], linewidth=0.2, antialiased=True)
    plt.show()


if __name__ == "__main__":
    main()