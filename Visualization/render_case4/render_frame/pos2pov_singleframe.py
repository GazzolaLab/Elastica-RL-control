import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


# pos=np.load("position_bend2.npy")
# print(pos.shape) #(time,x,n_elem)

# two_arm_pos=np.load("two_arm_data.npz")
# # print("pos_rod1",pos_rod1.shape) #(time,x,n_elem)
# position_rod1 = two_arm_pos["position_rod1"]
# radius_rod1 = two_arm_pos["radii_rod1"]
# n_elems_rod1 = two_arm_pos["n_elems_rod1"]
#
# position_rod2 = two_arm_pos["position_rod2"]
# radius_rod2 = two_arm_pos["radii_rod2"]
# n_elems_rod2 = two_arm_pos["n_elems_rod2"]
#
# position_rod3 = two_arm_pos["position_rod3"]
# radius_rod3 = two_arm_pos["radii_rod3"]
# n_elems_rod3= two_arm_pos["n_elems_rod3"]

# Load elastic arm data
arm_and_target_data = np.load("../data/arm_data.npz")
arm_position = arm_and_target_data["position_rod"]
arm_radius = arm_and_target_data["radii_rod"]
arm_n_elem = arm_and_target_data["n_elems_rod"]

# Load target data
target_radius = arm_and_target_data["radii_sphere"]
target_position = arm_and_target_data["position_sphere"]

# Load obstacle data
obstacle_data = np.load("../data/obstacle_data.npz", allow_pickle=True)

for k in obstacle_data:
    print(k)
obstacle_history = obstacle_data["obstacle_history"]


number_of_obstacles = len(obstacle_history)
number_of_elem_obstacles = np.array(obstacle_history[0]["position_plotting"]).shape[1]

# Obstacle positions are not changing throught the time
obstacle_radius = np.zeros((number_of_obstacles))
obstacle_position = np.zeros((3,number_of_elem_obstacles, number_of_obstacles))

for i in range(number_of_obstacles):
    obstacle_radius[i] = obstacle_history[i]["radius"]
    obstacle_position[...,i] = obstacle_history[i]["position_plotting"]


# activation_and_forces = np.load("two_arm_activation.npz")
# forces_rod1=activation_and_forces["force_muscle1"]
# # print("tor1",tor1.shape) #(time,x,n_elem)
# act1=activation_and_forces["activation_rod1"]
# print("act1",act1.shape) #(time,x,n_elem)
#
# forces_rod2 =  activation_and_forces["force_muscle2"]
# act2 = activation_and_forces["activation_rod2"]
#
#
# forces_rod3=activation_and_forces["force_muscle3"]
# # print("tor2",tor2.shape) #(time,x,n_elem)
# act3=activation_and_forces["activation_rod3"]
# print("act2",act2.shape) #(time,x,n_elem)
# radius = np.load("radius_of_arm.npy")
# print("radius",radius.shape)

scale=3 #simulation is 0.3m , other arms are 1m
time_elapse=False
video=True
combine=False
all_octopus=False
plot_muscle=False

time = arm_position.shape[0]


if time_elapse==True:
    file1 = open("moving_arm.inc","w")
    # for k in range(time):
    #     file1.writelines("sphere_sweep\n{b_spline %d"% n_elem)
    #     for i in range(n_elem):
    #         file1.writelines(",\n<%f,%f,%f>,%f"% (x[k][i],y[k][i],z[k][i],radius[k]))
    #     file1.writelines("\ntexture{")
    #     file1.writelines("pigment{ color rgb<0.45,0.39,1> transmit %f }"%transmit[k])
    #     file1.writelines("finish{ phong 1 } }")
    #     file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")
elif video==True:
    for k in range(1):
        file1 = open("images/moving_arm%02d.inc"%k,"w")

        # Arm
        file1.writelines("sphere_sweep\n{b_spline %d"% int(arm_n_elem))
        for i in range(0,arm_n_elem):
            # print(arm_position[k][0][i] , arm_position[k][1][i], arm_position[k][2][i],)
            # file1.writelines(",\n<%f,%f,%f>,%f"% (x[k][i]*scale,y[k][i]*scale,z[k][i]*scale,radius[k][1][i]*scale))
            file1.writelines(
                ",\n<%f,%f,%f>,%f" % (scale*arm_position[k][0][i] , scale*arm_position[k][1][i], scale*arm_position[k][2][i], scale*arm_radius[k][i]))
        file1.writelines("\ntexture{")
        file1.writelines("pigment{ color rgb<0.45,0.39,1>  transmit %f }"%(0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # Base of arm
        file1.writelines("sphere\n{")
        for i in range(0, 1):
            file1.writelines(
                "\n<%f,%f,%f>,%f" % (scale*arm_position[k][0][i] , scale*arm_position[k][1][i], scale*arm_position[k][2][i], 1.5 * scale * arm_radius[k][i]))
        file1.writelines("\ntexture{")
        file1.writelines("pigment{ color rgb<0.75,0.75,0.75>  transmit %f }" % (0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # Target
        file1.writelines("sphere\n{")
        file1.writelines(
            "\n<%f,%f,%f>,%f" % (
            scale * target_position[k][0], scale * (target_position[k][1]), scale * target_position[k][2],
            scale * target_radius[k]))
        file1.writelines("\ntexture{")
        file1.writelines("pigment{ color rgb<1,0.5,0.4>  transmit %f }" % (0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # Obstacles
        for j in range(0, number_of_obstacles):
            file1.writelines("sphere_sweep\n{b_spline %d" % int(number_of_elem_obstacles))
            for i in range(0, number_of_elem_obstacles):
                file1.writelines(
                    ",\n<%f,%f,%f>,%f" % (
                    scale * obstacle_position[0][i][j], scale * obstacle_position[1][i][j], scale * obstacle_position[2][i][j],
                    scale * obstacle_radius[j]))
            file1.writelines("\ntexture{")
            file1.writelines("pigment{ color LimeGreen  transmit %f }" % (0.1))
            file1.writelines("finish{ phong 1 } }")
            file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        file2 = open("images/moving_arm%02d.pov"%k,"w")
        file2.writelines("#include \"../snake.inc\"\n")
        file2.writelines("#include \"moving_arm%02d.inc\"\n"%k)

else:
    # the fps value has to change according to time_elapse (flag r)
    # os.system("ffmpeg -threads 8 -r 20 -i moving_arm%02d.png -b:v 90M -vcodec mpeg4 ./0.mp4")
    os.system(
        "ffmpeg -threads 8 -r 20 -i ./high_resolution/moving_arm%02d.png -b:v 90M -vcodec mpeg4 ./high_resolution/0.mp4")
