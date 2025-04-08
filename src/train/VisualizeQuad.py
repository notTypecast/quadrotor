import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from time import sleep

FILENAME = "data/quad.txt"
LEN_OFFSET = 1

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

quadrotor = []

def visualize(pos, quat, vel=None):

    for obj in quadrotor:
        obj.remove()

    quadrotor.clear()

    center = np.array(pos)
    points = np.array([
        [pos[0] - LEN_OFFSET, pos[1], pos[2]],
        [pos[0] + LEN_OFFSET, pos[1], pos[2]],
        [pos[0], pos[1] - LEN_OFFSET, pos[2]],
        [pos[0], pos[1] + LEN_OFFSET, pos[2]]
    ])

    upper_points = points + np.array((0, 0, LEN_OFFSET / 4))

    full_points = np.vstack((points, upper_points))

    rel_points = full_points - center

    r = R.from_quat(quat, scalar_first=True)
    rotated_rel_points = r.apply(rel_points)

    rotated_points = rotated_rel_points + center

    line1, = ax.plot3D([rotated_points[0][0], rotated_points[1][0]], [rotated_points[0][1], rotated_points[1][1]], [rotated_points[0][2], rotated_points[1][2]], c=(0.7, 0.2, 0.3), linewidth=4)
    line2, = ax.plot3D([rotated_points[2][0], rotated_points[3][0]], [rotated_points[2][1], rotated_points[3][1]], [rotated_points[2][2], rotated_points[3][2]], c=(0.7, 0.2, 0.3), linewidth=4)

    if vel is not None:
        vel = np.array(vel)
        vel = r.apply(vel)

    for i in range(4):        
        line, = ax.plot3D([rotated_points[i][0], rotated_points[i + 4][0]], [rotated_points[i][1], rotated_points[i + 4][1]], [rotated_points[i][2], rotated_points[i + 4][2]], c=(0.7, 0.2, 0.3), linewidth=4)
        quadrotor.append(line)
        point = ax.scatter(*rotated_points[i + 4], color=(0.7, 0.2, 0.3), s=20)
        quadrotor.append(point)

        if vel is not None:
            vec = ax.quiver(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], arrow_length_ratio=0.1)
            quadrotor.append(vec)

    quadrotor.extend([line1, line2])


prev_target = None

with open(FILENAME, "r") as f:
    f.seek(0, 2)

    while True:
        data = f.read()

        if data:
            # if multiple lines, ignore all but last
            data = [x for x in data.split("\n") if x]

            for piece in data:
                if piece.startswith("TARGET"):
                    piece = [float(x) for x in piece[6:].split()]

                    if prev_target is not None:
                        prev_target.remove()

                    prev_target = ax.scatter(*piece, color=(0.3, 0.6, 0.4), s=50)

                else:    
                    piece = piece.split()
                    state = [float(x) for x in piece[:7]]
                    vel = [float(x) for x in piece[7:10]]

                    pos = state[:3]
                    quat = state[3:]

                    visualize(pos, quat, vel)
            
        plt.pause(0.1)
