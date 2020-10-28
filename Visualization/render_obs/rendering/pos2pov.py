import numpy as np
import matplotlib.pyplot as plt
import os

# from nengo_extras.plot_spikes import (
#     cluster,
#     merge,
#     plot_spikes,
#     preprocess_spikes,
#     sample_by_variance,
#     sample_by_activity,
# )

# pos=np.load("position_bend2.npy")
# print(pos.shape) #(time,x,n_elem)

two_arm_pos=np.load("../data/arm_data.npz")
dir_id = "./images"
# print("pos_rod1",pos_rod1.shape) #(time,x,n_elem)
position_rod1 = two_arm_pos["position_rod"]
radius_rod1 = two_arm_pos["radii_rod"]
n_elems_rod1 = two_arm_pos["n_elems_rod"]

# radius_rod1=np.flip(np.arange(n_elems_rod1)*((0.05-0.01)/(n_elems_rod1-1))+0.015)

print(two_arm_pos["radii_rod"], radius_rod1)
print(n_elems_rod1, 0.5/(np.arange(n_elems_rod1)+1))

position_sphere = two_arm_pos["position_sphere"]
radius_sphere = two_arm_pos["radii_sphere"]

print(position_sphere.shape, radius_sphere.shape)

# spike_raster_data = np.load("inverted_pendulum_spikes.npz")
# spike_time_hist = spike_raster_data["time"]
# spikes = spike_raster_data["spikes"]

scale=3 #scales scene. Set at 3 to avoid resetting camera angles. 
video=True
combine=False
all_octopus=False
plot_muscle=False

#snap shot every (time_inc) frames
time_inc=0.1#int(pos.shape[0]/5)
# pos=pos[::time_inc,:,:]

# take the first few snap shots

time = position_rod1.shape[0]

if video==True:
    for k in range(time):
        file1 = open("%s/moving_arm%02d.inc" % (dir_id, k),"w")

        # Arm
        file1.writelines("sphere_sweep\n{b_spline %d"% int(n_elems_rod1))
        radius_scale = 1.0 #np.flip(np.linspace(0.25,1.0,n_elems_rod1))
        for i in range(0,n_elems_rod1):
            file1.writelines(
                ",\n<%f,%f,%f>,%f" % (scale*position_rod1[k][2][i] , scale*(position_rod1[k][0][i]), scale*position_rod1[k][1][i], scale*radius_rod1[k][i]*radius_scale))
        file1.writelines("\ntexture{")
        # file1.writelines("pigment{ color Orange transmit %f }"%(0.1))
        file1.writelines("pigment{ color rgb<0.45,0.39,1>  transmit %f }" % (0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")

        # Base of arm
        file1.writelines("sphere\n{")
        for i in range(0,1):
            # file1.writelines(",\n<%f,%f,%f>,%f"% (x[k][i]*scale,y[k][i]*scale,z[k][i]*scale,radius[k][1][i]*scale))
            file1.writelines(
                "\n<%f,%f,%f>,%f" % (scale*position_rod1[k][2][i]*0 , scale*(position_rod1[k][0][i])*0, scale*position_rod1[k][1][i]*0, 1.5*scale*radius_rod1[k][i]))
        file1.writelines("\ntexture{")
        file1.writelines("pigment{ color rgb<0.75,0.75,0.75>  transmit %f }"%(0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")
        
        # Target
        file1.writelines("sphere\n{")
        for i in range(0,1):
            # file1.writelines(",\n<%f,%f,%f>,%f"% (x[k][i]*scale,y[k][i]*scale,z[k][i]*scale,radius[k][1][i]*scale))
            file1.writelines(
                "\n<%f,%f,%f>,%f" % (scale*position_sphere[k][2][i] , scale*(position_sphere[k][0][i]), scale*position_sphere[k][1][i], scale*radius_sphere[k]))
        file1.writelines("\ntexture{")
        file1.writelines("pigment{ color rgb<1,0.5,0.4>  transmit %f }"%(0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines("scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n")
        
        file2 = open("%s/moving_arm%02d.pov" % (dir_id, k) ,"w")
        file2.writelines("#include \"../snake.inc\"\n")
        file2.writelines("#include \"moving_arm%02d.inc\"\n"%k)
else:
    # the fps value has to change according to time_elapse (flag r)
    os.system("ffmpeg -threads 8 -r 20 -i images/moving_arm%02d.png -b:v 90M -vcodec mpeg4 ./test.mp4")
