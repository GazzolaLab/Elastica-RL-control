import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from tqdm import tqdm

from typing import Dict, Sequence



two_arm_pos=np.load("./data/arm_data.npz")
two_arm_pos_no_obs=np.load("./data/arm_data_without_obstacles.npz")

for k in two_arm_pos.files:
    print(k)

# position_rod = two_arm_pos["position_rod"]
# radius_rod = two_arm_pos["radii_rod"]
# n_elems_rod = two_arm_pos["n_elems_rod"]
# position_sphere = two_arm_pos["position_sphere"]
# radius_sphere = two_arm_pos["radii_sphere"]

import matplotlib.animation as animation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d, Axes3D
from itertools import chain
from matplotlib.patches import Circle
import copy


fig = plt.figure(2, figsize=(10, 8), frameon=True)
# ax = fig.add_subplot(111)
ax = fig.add_subplot(111)


total = len(two_arm_pos["position_rod"][:])
total = 250
skip = 20




plot_idx = chain(range(0,50,5),range(50,135,20), range(135,220,35), range(220,300,20))
plot_idx_check = copy.deepcopy(plot_idx)
for total, _ in enumerate(plot_idx_check):
    pass

plot_idx_check_no_obs = chain(range(0,150,5))
for k, i in enumerate(plot_idx_check_no_obs):
    print(k/(total))
    pos = two_arm_pos_no_obs["position_rod"][i]
    # ax.plot(-pos[0],pos[2],'-', color = 'chocolate')#alpha = np.clip(k/(total), 0.5,1))

    pos = two_arm_pos["position_rod"][i]
    # ax.plot(-pos[0],pos[2],'-', color=(0.45,0.39,1))#, alpha = np.clip(k/(total), 0.2,1))
    ax.plot(-pos[0],pos[2],'-', color='k',alpha=1.0)#, alpha = np.clip(k/(total), 0.5,1))

# for k, i in enumerate(plot_idx):
# # for k, i in enumerate(range(1, int(total/skip)+00)):
#     pos = two_arm_pos["position_rod"][i]
#     # ax.plot(-pos[0],pos[2],'-', color=(0.45,0.39,1))#, alpha = np.clip(k/(total), 0.2,1))
#     ax.plot(-pos[0],pos[2],'-', color='k')#, alpha = np.clip(k/(total), 0.5,1))


# ax.plot(-two_arm_pos["position_rod"][:,0,-1],two_arm_pos["position_rod"][:,2,-1],'-', color='g',alpha=1.0)

target_pos = two_arm_pos["position_sphere"][0]
ax.plot(-target_pos[0],target_pos[2],'ro', alpha = 1.0)


# Load obstacle data
obstacle_data = np.load("./data/obstacle_data.npz", allow_pickle=True)
obstacle_history = obstacle_data["obstacle_history"]
number_of_obstacles = len(obstacle_history)
number_of_elem_obstacles = np.array(obstacle_history[0]["position_plotting"]).shape[1]
# Obstacle positions are not changing throught the time
obstacle_radius = np.zeros((number_of_obstacles))
obstacle_position = np.zeros((3,number_of_elem_obstacles, number_of_obstacles))
for i in range(number_of_obstacles):
    obstacle_radius[i] = obstacle_history[i]["radius"]
    obstacle_position[...,i] = obstacle_history[i]["position_plotting"]
    # ax.plot(-obstacle_position[...,i][0,0],obstacle_position[...,i][2,0],'o')
    circle = Circle((-obstacle_position[...,i][0,0],obstacle_position[...,i][2,0]),
        obstacle_radius[i], facecolor='w', edgecolor='g', alpha=1.0)
    ax.add_artist(circle)

plt.ylim([-0.025,1.1])
ax.set_aspect("equal")
plt.axis("off")
# plt.savefig("obstacles_cropped.png", dpi=600, transparent=True)
plt.show()


# plot_video_with_sphere_2D(
#     [self.post_processing_dict_rod],
#     [self.post_processing_dict_sphere],
#     video_name="2d_"+filename_video,
#     fps=self.rendering_fps,
#     step=1,
#     vis2D=False,
#     **kwargs,
# )

# plot_video_with_sphere(
#     [self.post_processing_dict_rod],
#     [self.post_processing_dict_sphere],
#     video_name="3d_"+filename_video,
#     fps=self.rendering_fps,
#     step=1,
#     vis2D=False,
#     **kwargs,
# )

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    # plt.close(plt.gcf())


