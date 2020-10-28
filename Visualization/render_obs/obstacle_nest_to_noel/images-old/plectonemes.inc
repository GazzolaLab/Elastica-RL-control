#version 3.6; // 3.7;
global_settings{ assumed_gamma 1.0 }
#default{ finish{ ambient 0.1 diffuse 0.9 }}

// INCLUDES
//--------------------------------------------------------------------------
#include "colors.inc"
#include "textures.inc"
#include "glass.inc"
#include "metals.inc"
#include "golds.inc"
#include "stones.inc"
#include "woods.inc"
#include "shapes.inc"
#include "shapes2.inc"
#include "functions.inc"
#include "math.inc"
#include "transforms.inc"

// CAMERA
//--------------------------------------------------------------------------
#declare Camera_Number = 1 ;
#declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
#declare Camera_Position = <-5.00, 5.00,-10.00> ;  // front view
#declare Camera_Look_At  = <2,0,0> ;
#declare Camera_Angle    =  20 ;
#declare Camera_Rotate = <0,0,0>; // tilling of the camera!!

camera{ location Camera_Position
        right    x*image_width/image_height
        angle    Camera_Angle
        look_at  Camera_Look_At
      }

light_source{< 4500,4500,-4500> color White}           // sun light
light_source{ Camera_Position  color rgb<0.9,0.9,1>*0.1}  // flash light




// GROUND
//--------------------------------------------------------------------------
#declare RasterScale = 2.0;
#declare RasterHalfLine  = 0.03;
#declare RasterHalfLineZ = 0.03;

#macro Raster(RScale, HLine)
       pigment{ gradient x scale RScale
                color_map{[0.000   color rgbt<1,1,1,0>*0.6]
                          [0+HLine color rgbt<1,1,1,0>*0.6]
                          [0+HLine color rgbt<1,1,1,1>]
                          [1-HLine color rgbt<1,1,1,1>]
                          [1-HLine color rgbt<1,1,1,0>*0.6]
                          [1.000   color rgbt<1,1,1,0>*0.6]} }
 #end// of Raster(RScale, HLine)-macro


plane { <0,1,0>, 0    // plane with layered textures
        texture { pigment{color White*1.1}
                  finish {ambient 0.45 diffuse 0.85}}
        texture { Raster(RasterScale,RasterHalfLine ) rotate<0,0,0> }
        texture { Raster(RasterScale,RasterHalfLineZ) rotate<0,90,0>}
        rotate<0,0,0>
      }
