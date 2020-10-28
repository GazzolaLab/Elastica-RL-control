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
arm_and_target_data = np.load("arm_data.npz")
arm_position = arm_and_target_data["position_rod"]
arm_radius = arm_and_target_data["radii_rod"]
arm_n_elem = arm_and_target_data["n_elems_rod"]

# Load target data
target_radius = arm_and_target_data["radii_sphere"]
target_position = arm_and_target_data["position_sphere"]

# Load obstacle data
obstacle_data = np.load("obstacle_data.npz", allow_pickle=True)
obstacle_history = obstacle_data["obstacle_history"]

number_of_obstacles = len(obstacle_history)
number_of_elem_obstacles = np.array(obstacle_history[0]["position_plotting"]).shape[1]

# Obstacle positions are not changing throught the time
obstacle_radius = np.zeros((number_of_obstacles))
obstacle_position = np.zeros((3,number_of_elem_obstacles, number_of_obstacles))

for i in range(number_of_obstacles):
    obstacle_radius[i] = obstacle_history[i]["radius"]
    obstacle_position[...,i] = obstacle_history[i]["position_plotting"]

obstacles_to_be_plotted = [0, 1, 3]

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
    for k in range(time):
        file1 = open("images/moving_arm%02d.inc"%k,"w")

        # Arm
        file1.writelines("sphere_sweep\n{b_spline %d"% int(arm_n_elem))
        for i in range(0,arm_n_elem):
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
                "\n<%f,%f,%f>,%f" % (scale * arm_position[k][0][i] , scale * (arm_position[k][1][i]) ,
                                     scale * arm_position[k][2][i] , 1.5 * scale * arm_radius[k][i]))
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
        # for j in obstacles_to_be_plotted:
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

        # Round box
        # file1.writelines("object\n{")
        # file1.writelines("Round_Box\n(")
        # file1.writelines(
        #     "\n<%f,%f,%f>, <%f, %f, %f>, %f, %f)" % (
        #         scale * (obstacle_position[0][0][7]-obstacle_radius[7]), scale * (obstacle_position[1][0][7]-obstacle_radius[7]), scale*(obstacle_position[2][0][7]-obstacle_radius[7]),
        #         scale * (obstacle_position[0][-1][7]+obstacle_radius[7]), scale * (obstacle_position[1][-1][7]+obstacle_radius[7]), scale*(obstacle_position[2][-1][7]+obstacle_radius[7]),
        #         obstacle_radius[7]*1.5, 0))
        # file1.writelines("\ntexture{")
        # file1.writelines("pigment{ color rgb<1,0.5,0.4>  transmit %f }" % (0.1))
        # file1.writelines("finish{ phong 1 } }")
        # file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # file1.writelines("object\n{")
        # file1.writelines("Round_Box\n(")
        # file1.writelines(
        #     "\n<%f,%f,%f>, <%f, %f, %f>, %f, %f)" % (
        #         scale * (obstacle_position[0][0][6] - obstacle_radius[6]),
        #         scale * (obstacle_position[1][0][6] - obstacle_radius[6]),
        #         scale * (obstacle_position[2][0][6] - 3*obstacle_radius[6]),
        #         scale * (obstacle_position[0][-1][6] + obstacle_radius[6]),
        #         scale * (obstacle_position[1][-1][6] + obstacle_radius[6]),
        #         scale * (obstacle_position[2][-1][6] + 3*obstacle_radius[6]),
        #         obstacle_radius[6]*1.5, 0
        #     ))
        # file1.writelines("\ntexture{")
        # file1.writelines("pigment{ color LimeGreen  transmit %f }" % (0.1))
        # file1.writelines("finish{ phong 1 } }")
        # file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # file1.writelines("object\n{")
        # file1.writelines("Round_Box\n(")
        # file1.writelines(
        #     "\n<%f,%f,%f>, <%f, %f, %f>, %f, %f)" % (
        #         scale * (obstacle_position[0][0][2] - obstacle_radius[2]),
        #         scale * (obstacle_position[1][0][2] - obstacle_radius[2]),
        #         scale * (obstacle_position[2][0][2] - obstacle_radius[2]),
        #         scale * (obstacle_position[0][-1][2] + obstacle_radius[2]),
        #         scale * (obstacle_position[1][-1][2] + obstacle_radius[2]),
        #         scale * (obstacle_position[2][-1][2] + obstacle_radius[2]),
        #         obstacle_radius[2]*1.5, 0
        #     ))
        # file1.writelines("\ntexture{")
        # file1.writelines("pigment{ color rgb<1,0.5,0.4>  transmit %f }" % (0.1))
        # file1.writelines("finish{ phong 1 } }")
        # file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # # Rod 2
        # file1.writelines("sphere_sweep\n{b_spline %d"% int(n_elems_rod2))
        # for i in range(0,n_elems_rod2):
        #     # file1.writelines(",\n<%f,%f,%f>,%f"% (x[k][i]*scale,y[k][i]*scale,z[k][i]*scale,radius[k][1][i]*scale))
        #     file1.writelines(
        #         ",\n<%f,%f,%f>,%f" % (-scale*position_rod2[k][1][i] , scale*(position_rod2[k][2][i]), scale*position_rod2[k][0][i], scale*radius_rod2[k][i]))
        # file1.writelines("\ntexture{")
        # file1.writelines("pigment{ color rgb<0.45,0.39,1>  transmit %f }"%(0.1))
        # file1.writelines("finish{ phong 1 } }")
        # file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")
        #
        # # Rod 3
        # file1.writelines("sphere_sweep\n{b_spline %d" % int(n_elems_rod3))
        # for i in range(0, n_elems_rod3):
        #     # file1.writelines(",\n<%f,%f,%f>,%f"% (x[k][i]*scale,y[k][i]*scale,z[k][i]*scale,radius[k][1][i]*scale))
        #     file1.writelines(
        #         ",\n<%f,%f,%f>,%f" % (
        #         -scale * position_rod3[k][1][i], scale * (position_rod3[k][2][i]), scale * position_rod3[k][0][i],
        #         scale * radius_rod3[k][i]))
        # file1.writelines("\ntexture{")
        # file1.writelines("pigment{ color LimeGreen transmit %f }" % (0.1))
        # file1.writelines("finish{ phong 1 } }")
        # file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # if plot_muscle==True:
        #     sep=[70,130,150,190]
        #     color_list=['Orange','SlateBlue','LimeGreen']
        #     for k in range(time):
        #         # file1 = open("moving_arm%02d.inc"%k,"w")
        #         for m in range(2):
        #             file1.writelines("sphere_sweep\n{b_spline %d"% (int((sep[2*m+1]-sep[2*m])/10)+1))
        #             for i in range(sep[2*m],sep[2*m+1]+1,10):
        #                 print(i)
        #                 print("mus",mus_radius[i])
        #                 file1.writelines(",\n<%f,%f,%f>,%f"% (x[k][i],y[k][i],z[k][i],mus_radius[i]))
        #             file1.writelines("\ntexture{")
        #             file1.writelines("pigment{ color %s transmit %f }"%(color_list[m],transmit[-1]))
        #             file1.writelines("finish{ phong 1 } }")
        #             file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        file1.close()
        file2 = open("images/moving_arm%02d.pov"%k,"w")
        file2.writelines("#include \"snake.inc\"\n")
        file2.writelines("#include \"moving_arm%02d.inc\"\n"%k)
        file2.close()
    if all_octopus==True:
        file2.writelines("#include \"arm4.inc\"\n")
        file2.writelines("#include \"data_0000027.inc\"\n")
        file2.writelines("#include \"suckerSmall.inc\"\n")
        file2.writelines("#include \"suckerBig.inc\"\n")
        file2.writelines("#include \"arm4suckerSmall.inc\"\n")
        file2.writelines("#include \"arm4suckerBig.inc\"\n")
        file2.writelines("#include \"armsurface_cover.inc\"\n")
# elif combine==True:
#     for k in range(time):
#         image=plt.imread("./high_resolution/moving_arm%02d.png"%k)
#
#         fig,ax=plt.subplots()
#         kk=image[:,0,:].reshape(1080,1,3)
#         kk=np.repeat(kk,200,axis=1)
#         # for layer in range(200):
#         image=np.hstack((kk,image))
#         ax.imshow(image)
#
#         # plt.show()
#         # ax.set_xlim([0,2800])
#         ax.set_xlim([0,1550])
#         # ax.set_xlim([0,280])
#
#         ax.axis("off")
#
#         a = plt.axes([0.15, 0.7, .16, .1375])
#         # plt.eventplot([np.where(act1[k][:9]!=0)[0],np.where(act1[k][9:-1]!=0)[0]+9,np.where(act1[k][-1]>0.01)[0]+12],colors=["r","r","r"],linelengths=[0.2,0.4,0.6],lineoffsets=[0.05,0.45,1.05])
#         # plt.xlim([0,13])
#         plt.eventplot(
#             [np.where(act1[k][:] != 0)[0]],
#             color=colors["orange"])#, linelengths=[0.2, 0.4, 0.6], lineoffsets=[0.05, 0.45, 1.05])
#         plt.xlim([-1, 6])
#         # plt.eventplot(np.random.rand(1,n_elem)[0],colors="b",linelengths=1,lineoffsets=0.2)
#         # plt.eventplot(np.random.rand(1,n_elem)[0],colors='k',linelengths=0.5,lineoffsets=0.)
#         a.axes.get_xaxis().set_ticks([])
#         a.axes.get_yaxis().set_ticks([])
#
#         a = plt.axes([0.34, 0.7, .16, .1375])
#         # plt.eventplot([np.where(act3[k][:9]!=0)[0],np.where(act3[k][9:-1]!=0)[0]+9,np.where(act3[k][-1]>0.01)[0]+12],colors=["b","b","b"],linelengths=[0.2,0.4,0.6],lineoffsets=[0.05,0.45,1.05])
#         # plt.xlim([0,13])
#         plt.eventplot(
#             [np.where(act3[k][:] != 0)[0]],
#             color=colors["mediumpurple"])#, linelengths=[0.2, 0.4, 0.6], lineoffsets=[0.05, 0.45, 1.05])
#         plt.xlim([-1, 6])
#         a.axes.get_xaxis().set_ticks([])
#         a.axes.get_yaxis().set_ticks([])
#
#         a = plt.axes([0.53, 0.7, .16, .1375])
#         # plt.eventplot([np.where(act3[k][:9]!=0)[0],np.where(act3[k][9:-1]!=0)[0]+9,np.where(act3[k][-1]>0.01)[0]+12],colors=["b","b","b"],linelengths=[0.2,0.4,0.6],lineoffsets=[0.05,0.45,1.05])
#         # plt.xlim([0,13])
#         plt.eventplot(
#             [np.where(act2[k][:] != 0)[0]],
#             color=colors["limegreen"])#, linelengths=[0.2, 0.4, 0.6], lineoffsets=[0.05, 0.45, 1.05])
#         plt.xlim([-1, 6])
#         a.axes.get_xaxis().set_ticks([])
#         a.axes.get_yaxis().set_ticks([])
#
#         a = plt.axes([0.15, 0.45, .35, .2])
#         plt.plot(range(n_elems_rod1),forces_rod1[k],color = colors['orange'])
#         plt.plot(range(n_elems_rod1),forces_rod3[k],'-.', color=colors['mediumpurple'])
#         plt.plot(range(n_elems_rod1),forces_rod2[k]*0.5,'--', color=colors['limegreen'])
#         plt.ylim([0,10])
#         # plt.plot(range(n_elem),np.random.rand(1,n_elem)[0])
#         # a.axes.get_xaxis().set_ticks([])
#         a.axes.get_yaxis().set_ticks([])
#         # plt.title('Control Signal')
#
#         # a = plt.axes([0.15, 0.175, .35, .2])
#         # plt.plot(range(n_elem),np.random.rand(1,n_elem)[0])
#         # # a.axes.get_xaxis().set_ticks([])
#         # a.axes.get_yaxis().set_ticks([])
#         # plt.title('Torque')
#         # plt.show()
#         # fig,ax=plt.subplots()
#         # ax.plot(range(n_elem),np.random.rand(1,n_elem)[0])
#         plt.savefig("./high_resolution/combine_arm%02d.png"%k, dpi = 1000)
#         plt.close('all')
#     os.system("ffmpeg -threads 8 -r 20 -i ./high_resolution/combine_arm%02d.png -b:v 90M -vcodec mpeg4 ./high_resolution/0.mp4")

else:
    # the fps value has to change according to time_elapse (flag r)
    os.system("ffmpeg -threads 8 -r 20 -i moving_arm%02d.png -b:v 90M -vcodec mpeg4 ./0.mp4")

