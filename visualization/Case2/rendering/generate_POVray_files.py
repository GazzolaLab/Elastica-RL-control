import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Load elastic arm data
arm_and_target_data = np.load("../data/arm_data.npz")
arm_position = arm_and_target_data["position_rod"]
arm_radius = arm_and_target_data["radii_rod"]
arm_n_elem = arm_and_target_data["n_elems_rod"]

# Load target data
target_radius = arm_and_target_data["radii_sphere"]
target_position = arm_and_target_data["position_sphere"]

# Plot directors for rod tip and sphere
arm_directors = arm_and_target_data["directors_rod"]
target_directors = arm_and_target_data["directors_sphere"]
vector_radius = 0.025 # Radius of lines
color_order = ["SlateBlue", "LimeGreen", "OrangeRed"]

scale=3 #simulation is 0.3m , other arms are 1m

time = arm_position.shape[0]

for k in range(time):
    file1 = open("images/moving_arm%02d.inc"%k,"w")

    # Arm
    file1.writelines("sphere_sweep\n{b_spline %d"% int(arm_n_elem))
    for i in range(0,arm_n_elem):
        file1.writelines(
            ",\n<%f,%f,%f>,%f" % (scale*arm_position[k][2][i] , scale*arm_position[k][0][i], scale*arm_position[k][1][i], scale*arm_radius[k][i]))
    file1.writelines("\ntexture{")
    file1.writelines("pigment{ color rgb<0.45,0.39,1>  transmit %f }"%(0.1))
    file1.writelines("finish{ phong 1 } }")
    file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

    # Base of arm
    file1.writelines("sphere\n{")
    for i in range(0, 1):
        file1.writelines(
            "\n<%f,%f,%f>,%f" % (scale * arm_position[k][2][i] , scale * (arm_position[k][0][i]) ,
                                 scale * arm_position[k][1][i] , 1.5 * scale * arm_radius[k][i]))
    file1.writelines("\ntexture{")
    file1.writelines("pigment{ color rgb<0.75,0.75,0.75>  transmit %f }" % (0.1))
    file1.writelines("finish{ phong 1 } }")
    file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

    # Target
    file1.writelines("sphere\n{")
    file1.writelines(
        "\n<%f,%f,%f>,%f" % (
        scale * target_position[k][2], scale * (target_position[k][0]), scale * target_position[k][1],
        scale * target_radius[k]))
    file1.writelines("\ntexture{")
    file1.writelines("pigment{ color rgb<1,0.5,0.4>  transmit %f }" % (0.1))
    file1.writelines("finish{ phong 1 } }")
    file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

    # # Obstacles
    # for j in range(0, number_of_obstacles):
    # # for j in obstacles_to_be_plotted:
    #     file1.writelines("sphere_sweep\n{b_spline %d" % int(number_of_elem_obstacles))
    #     for i in range(0, number_of_elem_obstacles):
    #         file1.writelines(
    #             ",\n<%f,%f,%f>,%f" % (
    #             scale * obstacle_position[0][i][j], scale * obstacle_position[1][i][j], scale * obstacle_position[2][i][j],
    #             scale * obstacle_radius[j]))
    #     file1.writelines("\ntexture{")
    #     file1.writelines("pigment{ color LimeGreen  transmit %f }" % (0.1))
    #     file1.writelines("finish{ phong 1 } }")
    #     file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

    # Drawing rod tip directors using vector
    for idx in range(3):
        arm_tip_vector_start = arm_position[k, :,-1]
        arm_tip_vector_end = arm_position[k, :, -1] + 5*arm_radius[k,-1] * arm_directors[k, idx, :, -1] # directors(time, vectorstart, vectorend, element index)
        file1.writelines("object\n{")
        file1.writelines("Vector\n(")
        file1.writelines(
            "\n<%f,%f,%f>,   <%f, %f, %f>, %f)" % (
                scale * (arm_tip_vector_start[2]),
                scale * (arm_tip_vector_start[0]),
                scale * (arm_tip_vector_start[1]),
                scale * (arm_tip_vector_end[2] ),
                scale * (arm_tip_vector_end[0] ),
                scale * (arm_tip_vector_end[1] ),
                vector_radius,
                ))
        file1.writelines("\ntexture{")
        file1.writelines(str("pigment{ color "+color_order[idx]+" transmit %f }") % (0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

    # Drawing target directors using vector
    for idx in range(3):
        target_vector_start = target_position[k, :, -1]
        target_vector_end = target_position[k, :, -1] + 5 * target_radius[k, -1] * target_directors[k, idx, :,
                                                                        -1]  # directors(time, vectorstart, vectorend, element index)
        file1.writelines("object\n{")
        file1.writelines("Vector\n(")
        file1.writelines(
            "\n<%f,%f,%f>,   <%f, %f, %f>, %f)" % (
                scale * (target_vector_start[2]),
                scale * (target_vector_start[0]),
                scale * (target_vector_start[1]),
                scale * (target_vector_end[2]),
                scale * (target_vector_end[0]),
                scale * (target_vector_end[1]),
                vector_radius,
            ))
        file1.writelines("\ntexture{")
        file1.writelines(str("pigment{ color "+color_order[idx]+" transmit %f }") % (0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

    file1.close()
    file2 = open("images/moving_arm%02d.pov" % k, "w")
    file2.writelines("#include \"../camera_position.inc\"\n")
    file2.writelines("#include \"moving_arm%02d.inc\"\n" % k)
    file2.close()


