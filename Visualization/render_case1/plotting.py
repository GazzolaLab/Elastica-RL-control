import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from tqdm import tqdm

from typing import Dict, Sequence


def plot_video_with_sphere(
    rods_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    # simulation time
    sim_time = np.array(sphere_history[0]["time"])

    # Sphere
    n_visualized_spheres = len(sphere_history)  # should be one for now
    # Sphere radius
    sphere_radii = [x["radius"] for x in sphere_history]
    # Sphere info
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )
    # color mapping
    sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.0, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    # ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    sphere_history_directors = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["directors"][t_idx],
    )
    rod_history_directors = lambda sph_idx, t_idx: (
        rods_history[sph_idx]["directors"][t_idx],
    )
    rod_directors1 = [None for _ in range(n_visualized_rods)]
    rod_directors2 = [None for _ in range(n_visualized_rods)]
    rod_directors3 = [None for _ in range(n_visualized_rods)]
    sphere_directors1 = [None for _ in range(n_visualized_spheres)]
    sphere_directors2 = [None for _ in range(n_visualized_spheres)]
    sphere_directors3 = [None for _ in range(n_visualized_spheres)]

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        # rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1],inst_position[2], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        # rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1],inst_com[2], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[2],
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

        rod_dir = rod_history_directors(rod_idx, time_idx) 
        scale = 0.2
        dir1 = rod_dir[0][...,-1][0] * scale
        dir2 = rod_dir[0][...,-1][1] * scale
        dir3 = rod_dir[0][...,-1][2] * scale
        tip_position = inst_position[..., -1]
        rod_directors1[rod_idx] = ax.plot(
            np.array([tip_position[2], tip_position[2] + dir1[2]]).flatten(),
            np.array([tip_position[0], tip_position[0] + dir1[0]]).flatten(),
            np.array([tip_position[1], tip_position[1] + dir1[1]]).flatten(),
            "r-", lw=2.0)[0]
        rod_directors2[rod_idx] = ax.plot(
            np.array([tip_position[2], tip_position[2] + dir2[2]]).flatten(),
            np.array([tip_position[0], tip_position[0] + dir2[0]]).flatten(),
            np.array([tip_position[1], tip_position[1] + dir2[1]]).flatten(),
            "g-", lw=2.0)[0]
        rod_directors3[rod_idx] = ax.plot(
            np.array([tip_position[2], tip_position[2] + dir3[2]]).flatten(),
            np.array([tip_position[0], tip_position[0] + dir3[0]]).flatten(),
            np.array([tip_position[1], tip_position[1] + dir3[1]]).flatten(),
            "b-", lw=2.0)[0]


    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = ax.scatter(
            sphere_position[2],
            sphere_position[0],
            sphere_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )
        # sphere_radius,
        # color=sphere_cmap(sphere_idx),)
        ax.add_artist(sphere_artists[sphere_idx])

        sphere_dir = sphere_history_directors(sphere_idx, time_idx) 
        scale = 0.2
        dir1 = sphere_dir[0][0] * scale
        dir2 = sphere_dir[0][1] * scale
        dir3 = sphere_dir[0][2] * scale
        sphere_directors1[sphere_idx] = ax.plot(
            np.array([sphere_position[0], sphere_position[0] + dir1[0]]).flatten(),
            np.array([sphere_position[1], sphere_position[1] + dir1[1]]).flatten(),
            np.array([sphere_position[2], sphere_position[2] + dir1[2]]).flatten(),
            "r-", lw=2.0)[0]
        sphere_directors2[sphere_idx] = ax.plot(
            np.array([sphere_position[0], sphere_position[0] + dir2[0]]).flatten(),
            np.array([sphere_position[1], sphere_position[1] + dir2[1]]).flatten(),
            np.array([sphere_position[2], sphere_position[2] + dir2[2]]).flatten(),
            "g-", lw=2.0)[0]
        sphere_directors3[sphere_idx] = ax.plot(
            np.array([sphere_position[0], sphere_position[0] + dir3[0]]).flatten(),
            np.array([sphere_position[1], sphere_position[1] + dir3[1]]).flatten(),
            np.array([sphere_position[2], sphere_position[2] + dir3[2]]).flatten(),
            "b-", lw=2.0)[0]

    # ax.set_aspect("equal")
    video_name = "2D_" + video_name

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    # rod_lines[rod_idx].set_xdata(inst_position[0])
                    # rod_lines[rod_idx].set_ydata(inst_position[1])
                    # rod_lines[rod_idx].set_zdata(inst_position[2])

                    com = com_history_unpacker(rod_idx, time_idx)
                    # rod_com_lines[rod_idx].set_xdata(com[0])
                    # rod_com_lines[rod_idx].set_ydata(com[1])
                    # rod_com_lines[rod_idx].set_zdata(com[2])

                    # rod_scatters[rod_idx].set_offsets(inst_position[:3].T)
                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[2],
                        inst_position[0],
                        inst_position[1],
                    )

                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2 * 0.1
                    )

                    rod_dir = rod_history_directors(sphere_idx, time_idx) 
                    dir1 = rod_dir[0][...,-1][0] * scale
                    dir2 = rod_dir[0][...,-1][1] * scale
                    dir3 = rod_dir[0][...,-1][2] * scale
                    tip_position = inst_position[..., -1]
                    rod_directors1[rod_idx].set_xdata(np.array([tip_position[2], tip_position[2] + dir1[2]]).flatten())
                    rod_directors1[rod_idx].set_ydata(np.array([tip_position[0], tip_position[0] + dir1[0]]).flatten())
                    rod_directors1[rod_idx].set_3d_properties(np.array([tip_position[1], tip_position[1] + dir1[1]]).flatten())

                    rod_directors2[rod_idx].set_xdata(np.array([tip_position[2], tip_position[2] + dir2[2]]).flatten())
                    rod_directors2[rod_idx].set_ydata(np.array([tip_position[0], tip_position[0] + dir2[0]]).flatten())
                    rod_directors2[rod_idx].set_3d_properties(np.array([tip_position[1], tip_position[1] + dir2[1]]).flatten())

                    rod_directors3[rod_idx].set_xdata(np.array([tip_position[2], tip_position[2] + dir3[2]]).flatten())
                    rod_directors3[rod_idx].set_ydata(np.array([tip_position[0], tip_position[0] + dir3[0]]).flatten())
                    rod_directors3[rod_idx].set_3d_properties(np.array([tip_position[1], tip_position[1] + dir3[1]]).flatten())



                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    # sphere_artists[sphere_idx].center = (
                    #     sphere_position[0],
                    #     sphere_position[1],
                    #     sphere_position[2],
                    # )
                    # sphere_artists[sphere_idx].set_offsets(sphere_position[:3].T)
                    sphere_artists[sphere_idx]._offsets3d = (
                        sphere_position[2],
                        sphere_position[0],
                        sphere_position[1],
                    )

                    sphere_dir = sphere_history_directors(sphere_idx, time_idx) 
                    dir1 = sphere_dir[0][0] * scale
                    dir2 = sphere_dir[0][1] * scale
                    dir3 = sphere_dir[0][2] * scale

                    sphere_directors1[sphere_idx].set_xdata(np.array([sphere_position[2], sphere_position[2] + dir1[2]]).flatten())
                    sphere_directors1[sphere_idx].set_ydata(np.array([sphere_position[0], sphere_position[0] + dir1[0]]).flatten())
                    sphere_directors1[sphere_idx].set_3d_properties(np.array([sphere_position[1], sphere_position[1] + dir1[1]]).flatten())

                    sphere_directors2[sphere_idx].set_xdata(np.array([sphere_position[2], sphere_position[2] + dir2[2]]).flatten())
                    sphere_directors2[sphere_idx].set_ydata(np.array([sphere_position[0], sphere_position[0] + dir2[0]]).flatten())
                    sphere_directors2[sphere_idx].set_3d_properties(np.array([sphere_position[1], sphere_position[1] + dir2[1]]).flatten())       

                    sphere_directors3[sphere_idx].set_xdata(np.array([sphere_position[2], sphere_position[2] + dir3[2]]).flatten())
                    sphere_directors3[sphere_idx].set_ydata(np.array([sphere_position[0], sphere_position[0] + dir3[0]]).flatten())
                    sphere_directors3[sphere_idx].set_3d_properties(np.array([sphere_position[1], sphere_position[1] + dir3[1]]).flatten())


                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_video_with_sphere_2D(
    rods_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    # simulation time
    sim_time = np.array(sphere_history[0]["time"])

    # Sphere
    n_visualized_spheres = len(sphere_history)  # should be one for now
    # Sphere radius
    sphere_radii = [x["radius"] for x in sphere_history]
    # Sphere info
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )

    # color mapping
    sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )

    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.1, 1.1))
    ylim = kwargs.get("y_limits", (-1.1, 1.1))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    sphere_history_directors = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["directors"][t_idx],
    )
    rod_history_directors = lambda sph_idx, t_idx: (
        rods_history[sph_idx]["directors"][t_idx],
    )
    rod_directors1 = [None for _ in range(n_visualized_rods)]
    rod_directors2 = [None for _ in range(n_visualized_rods)]
    rod_directors3 = [None for _ in range(n_visualized_rods)]
    sphere_directors1 = [None for _ in range(n_visualized_spheres)]
    sphere_directors2 = [None for _ in range(n_visualized_spheres)]
    sphere_directors3 = [None for _ in range(n_visualized_spheres)]

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]
    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

        rod_dir = rod_history_directors(rod_idx, time_idx) 
        scale = 0.2
        dir1 = rod_dir[0][...,-1][0] * scale
        dir2 = rod_dir[0][...,-1][1] * scale
        dir3 = rod_dir[0][...,-1][2] * scale
        tip_position = inst_position[..., -1]
        rod_directors1[rod_idx] = ax.plot(
            [tip_position[0], tip_position[0] + dir1[0]],
            [tip_position[1], tip_position[1] + dir1[1]],"r-", lw=2.0)[0]
        rod_directors2[rod_idx] = ax.plot(
            [tip_position[0], tip_position[0] + dir2[0]],
            [tip_position[1], tip_position[1] + dir2[1]],"g-", lw=2.0)[0]
        rod_directors3[rod_idx] = ax.plot(
            [tip_position[0], tip_position[0] + dir3[0]],
            [tip_position[1], tip_position[1] + dir3[1]],"b-", lw=2.0)[0]
        

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = Circle(
            (sphere_position[0], sphere_position[1]),
            sphere_radius,
            color=sphere_cmap(sphere_idx),
        )
        ax.add_artist(sphere_artists[sphere_idx])
        sphere_dir = sphere_history_directors(sphere_idx, time_idx) 
        scale = 0.2
        dir1 = sphere_dir[0][0] * scale
        dir2 = sphere_dir[0][1] * scale
        dir3 = sphere_dir[0][2] * scale
        sphere_directors1[sphere_idx] = ax.plot(
            [sphere_position[0], sphere_position[0] + dir1[0]],
            [sphere_position[1], sphere_position[1] + dir1[1]],"r-", lw=2.0)[0]
        sphere_directors2[sphere_idx] = ax.plot(
            [sphere_position[0], sphere_position[0] + dir2[0]],
            [sphere_position[1], sphere_position[1] + dir2[1]],"g-", lw=2.0)[0]
        sphere_directors3[sphere_idx] = ax.plot(
            [sphere_position[0], sphere_position[0] + dir3[0]],
            [sphere_position[1], sphere_position[1] + dir3[1]],"b-", lw=2.0)[0]

    ax.set_aspect("equal")
    video_name = "2D_" + video_name

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])

                    com = com_history_unpacker(rod_idx, time_idx)
                    rod_com_lines[rod_idx].set_xdata(com[0])
                    rod_com_lines[rod_idx].set_ydata(com[1])

                    rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )

                    rod_dir = rod_history_directors(sphere_idx, time_idx) 
                    dir1 = rod_dir[0][...,-1][0] * scale
                    dir2 = rod_dir[0][...,-1][1] * scale
                    dir3 = rod_dir[0][...,-1][2] * scale
                    tip_position = inst_position[..., -1]
                    rod_directors1[rod_idx].set_xdata([tip_position[0], tip_position[0] + dir1[0]])
                    rod_directors1[rod_idx].set_ydata([tip_position[1], tip_position[1] + dir1[1]])

                    rod_directors2[rod_idx].set_xdata([tip_position[0], tip_position[0] + dir2[0]])
                    rod_directors2[rod_idx].set_ydata([tip_position[1], tip_position[1] + dir2[1]])        

                    rod_directors3[rod_idx].set_xdata([tip_position[0], tip_position[0] + dir3[0]])
                    rod_directors3[rod_idx].set_ydata([tip_position[1], tip_position[1] + dir3[1]])

                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    sphere_artists[sphere_idx].center = (
                        sphere_position[0],
                        sphere_position[1],
                    )

                    sphere_dir = sphere_history_directors(sphere_idx, time_idx) 
                    dir1 = sphere_dir[0][0] * scale
                    dir2 = sphere_dir[0][1] * scale
                    dir3 = sphere_dir[0][2] * scale

                    sphere_directors1[sphere_idx].set_xdata([sphere_position[0], sphere_position[0] + dir1[0]])
                    sphere_directors1[sphere_idx].set_ydata([sphere_position[1], sphere_position[1] + dir1[1]])

                    sphere_directors2[sphere_idx].set_xdata([sphere_position[0], sphere_position[0] + dir2[0]])
                    sphere_directors2[sphere_idx].set_ydata([sphere_position[1], sphere_position[1] + dir2[1]])        

                    sphere_directors3[sphere_idx].set_xdata([sphere_position[0], sphere_position[0] + dir3[0]])
                    sphere_directors3[sphere_idx].set_ydata([sphere_position[1], sphere_position[1] + dir3[1]])
                writer.grab_frame()



two_arm_pos=np.load("./data/arm_data.npz")
for k in two_arm_pos.files:
    print(k)

position_rod = two_arm_pos["position_rod"]
radius_rod = two_arm_pos["radii_rod"]
n_elems_rod = two_arm_pos["n_elems_rod"]
position_sphere = two_arm_pos["position_sphere"]
radius_sphere = two_arm_pos["radii_sphere"]

import matplotlib.animation as animation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d, Axes3D

# video pre-processing
print("plot scene visualization video")



fig = plt.figure(2, figsize=(10, 8), frameon=True)
# ax = fig.add_subplot(111)
ax = fig.add_subplot(111, projection="3d")

# ax.plot()

print(len(two_arm_pos["position_rod"][:]))
total = len(two_arm_pos["position_rod"][:])
# total = 300
# skip = 5
# for k, i in enumerate(range(1, int(total/skip)+00)):
#     pos = two_arm_pos["position_rod"][i*skip]
#     target_pos = two_arm_pos["position_sphere"][i*skip]
#     ax.plot(pos[0],pos[1],pos[2],'k-', alpha = np.clip(k/(total/skip) + 0.2, 0,1))
#     ax.plot(target_pos[0],target_pos[1],target_pos[2],'ro',alpha = np.clip(k/(total/skip) + 0.2, 0,1))
# target_pos = (two_arm_pos["position_sphere"].T)[0]
# print(target_pos.shape)

total = len(two_arm_pos["position_rod"][:])
total = 100
skip = 1
start = [1, 59, 138, 323, 566]
color_list = ['c', 'k','r','b','g']
for k, i in enumerate(start):
    for j in range(10):
        pos = two_arm_pos["position_rod"][i+j*skip]
        target_pos = two_arm_pos["position_sphere"][i+j*skip]
        ax.plot(-pos[2],pos[0],pos[1],color_list[k]+'-', alpha = np.clip((10-j)/10, 0.2,1))
        ax.plot(-target_pos[2],target_pos[0],target_pos[1],color_list[k]+'o',alpha = np.clip((10-j)/10, 0.2,1))


ax.view_init(elev=15., azim=-150)
# plt.axis('off')
# plt.grid(b=None)

# plt.show()

print(np.array([1, 59, 138, 323, 566])/60)

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


