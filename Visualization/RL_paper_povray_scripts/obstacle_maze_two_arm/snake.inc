// POV-Ray 3.6 / 3.7 Scene File "Ribbon_Cable_1.pov"
// author: Friedrich A. Lohmueller, Sept-2009/Jan-2011
// email: Friedrich.Lohmueller_at_t-online.de
// homepage: http://www.f-lohmueller.de
//--------------------------------------------------------------------------
#version 3.6; // 3.7;
global_settings{ assumed_gamma 1.0 }
#default{ finish{ ambient 0.1 diffuse 0.9 }}
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
//--------------------------------------------------------------------------------------------------------<<<<
//------------------------------------------------------------- Camera_Position, Camera_look_at, Camera_Angle
	#declare Camera_Number = 17;
//----------------------------------------89----------------------------------------------------------------<<<<
#switch ( Camera_Number )
#case (0)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = < 7.00, 4.00,-10.00> ;  // Topside view close-Crab/ Walker
  #declare Camera_Look_At  = < 3.00, 1.00,  4.00> ;
  #declare Camera_Angle    =  55 ;    //75
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (1)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <23.00, 35.00,-37.00> ;  // Topside view — Nest
  #declare Camera_Look_At  = < 8.50, 19.000,  3.00> ;
  #declare Camera_Angle    =  107 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (2)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = < 4.00, 1.50,-4.00> ;  //side view
  #declare Camera_Look_At  = < 4.00, 0.50,  2.00> ;
  #declare Camera_Angle    =  70 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (3)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <2.0, 160.00,3.0> ;  // Top view —— Snake
  #declare Camera_Look_At  = <2.0, 0.00,3.0> ;
  #declare Camera_Angle    =  5;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (4)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = < 13.00, 3.00, -4.00> ;  //side view ——high— for elbow
  #declare Camera_Look_At  = < 0.500, 3,  5.00> ;
  #declare Camera_Angle    =  18 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (5)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <28.7, 15.00,80.8> ;  // Front view —Wings
  #declare Camera_Look_At  = < 28.7, 5.00,-0.8> ;
  #declare Camera_Angle    =  65;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (6)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <30, 300.00,-7.8> ;  // Top view —Wings
  #declare Camera_Look_At  = < 30, 3.00,-7.8> ;
  #declare Camera_Angle    =  15;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (7)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <158.7, 10.00,-0.8> ;  // Lateral view —Wings
  #declare Camera_Look_At  = < 28.7, 3.00,-0.8> ;
  #declare Camera_Angle    =  35;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (8)
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <4.0, 20.00,-5.5> ;  // Top view —Sperm
  #declare Camera_Look_At  = < 4.0, 0.00,-5.5> ;
  #declare Camera_Angle    =  130;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (9) //cross
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <6.0, 0.30,-1.5> ;  // Topside view
  #declare Camera_Look_At  = < 0.50, 1.700,  7.00> ;
  #declare Camera_Angle    =  28 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (10) //my octo
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <11.00, 6.00,-17.00> ;  // Topside view
  #declare Camera_Look_At  = < 0.50, 1.700,  7.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (11) //test
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <60.00, 1.00, -8.00> ;  // Topside view
  #declare Camera_Look_At  = < -360.50, 10.700,  20.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (12) //test Inverted pendulum
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <60.00, 1.00, -8.00> ;  // Topside view
  #declare Camera_Look_At  = < -3.50, 5.700,  20.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (13) //test Inverted pendulum
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <60.00, 1.00, -50.00> ;  // Topside view
  #declare Camera_Look_At  = < -3.50, 5.700,  40.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (14) //test Inverted pendulum
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <00.00, 1.00, -50.00> ;  // Topside view
  #declare Camera_Look_At  = < 0.0, 5.700,  10.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (15) //test 3D reaching arm
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <50.00, 10.00, -60.00> ;  // Topside view
  #declare Camera_Look_At  = < -3.50, 5.700,  10.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#break
#case (16) //test obstacle nest
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <50.00, 50.00, -70.00> ;  // Topside view
  #declare Camera_Look_At  = < -13.50, -15.700,  50.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
//#break
//#case (17) //test obstacle maze
//  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
//  #declare Camera_Position = <50.00, 50.00, 20.00> ;  // Topside view
//  #declare Camera_Look_At  = < -13.50, -25.700,  -20.00> ;
//  #declare Camera_Angle    =  30 ;
//  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
//#break
#break
#case (17) //test obstacle maze 3d view
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <70.00, 25.00, -20.00> ;  // Topside view
  #declare Camera_Look_At  = < -13.50, 5.700,  20.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#break
#case (18) //test obstacle maze orthogonal view
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <2200.00, 5.00, 5.00> ;  // Topside view
  #declare Camera_Look_At  = < 0.00, 0.000,  0.00> ;
  #declare Camera_Angle    =  1 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#else  //Xiaotian octo
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <23.00, 6.00,-18.00> ;  // Topside view
  #declare Camera_Look_At  = < 0.50, 1.700,  7.00> ;
  #declare Camera_Angle    =  28 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#end // of "#switch ( Camera_Number )" -----------------------------
//---------------------------------------------------------------------

camera{ location Camera_Position
        right    x*image_width/image_height
        angle    Camera_Angle
        look_at  Camera_Look_At
      }
//light_source{< 0,300,500> color White}
light_source{< 1500,1000,0>color White}           // sun light
light_source{ Camera_Position  color rgb<0.9,0.9,1>*0.1}  // flash light


/*
// fog ---------------------------------------------------------------
fog{fog_type   2   distance 65  color rgb<1,1,1>
fog_offset 0.1 fog_alt  2.0 turbulence 0.2}
//--------------------------------------------------------------------
*/
// <1,0,0>, -3
// <0,1,0>, 0
//<0,0,1>,-200
background { color White }

plane { <0,1.5,0>,-0.1  // plane with layered textures
texture { pigment{color White *1.5} // White*1.1
             }
}

//finish {ambient 0.45 diffuse 0.85}
/*
//------------------------------ the Axes --------------------------------
//------------------------------------------------------------------------
#macro Axis_( AxisLen, Dark_Texture,Light_Texture)
 union{
    cylinder { <0,-AxisLen,0>,<0,AxisLen,0>,0.05
               texture{checker texture{Dark_Texture }
                               texture{Light_Texture}
               scale 0.5
               translate<0.1,0,0.1>}
             }
    cone{<0,AxisLen,0>,0.2,<0,AxisLen+0.7,0>,0
          texture{Dark_Texture}
         }
     } // end of union
#end // of macro "Axis()"
//------------------------------------------------------------------------
#macro AxisXYZ( AxisLenX, AxisLenY, AxisLenZ, Tex_Dark, Tex_Light)
//--------------------- drawing of 3 Axes --------------------------------
union{
#if (AxisLenX != 0)
 object { Axis_(AxisLenX, Tex_Dark, Tex_Light)   rotate< 0,0,-90>}// x-Axis
#end // of #if
#if (AxisLenY != 0)
 object { Axis_(AxisLenY, Tex_Dark, Tex_Light)   rotate< 0,0,  0>}// y-Axis
#end // of #if
#if (AxisLenZ != 0)
 object { Axis_(AxisLenZ, Tex_Dark, Tex_Light)   rotate<90,0,  0>}// z-Axis
#end // of #if
} // end of union
#end// of macro "AxisXYZ( ... )"
//------------------------------------------------------------------------

#declare Texture_A_Dark  = texture {
                               pigment{ color rgb<1,0.4,0>}
                               finish { phong 1}
                             }
#declare Texture_A_Light = texture {
                               pigment{ color rgb<1,1,1>}
                               finish { phong 1}
                             }

object{ AxisXYZ( 2.70, 2.70, 2.70, Texture_A_Dark, Texture_A_Light) scale 0.75 }
//-------------------------------------------------- end of coordinate axes


// ground -----------------------------------------------------------------
//---------------------------------<<< settings of squared plane dimensions
#declare RasterScale = 2.0;
#declare RasterHalfLine  = 0.03;
#declare RasterHalfLineZ = 0.03;
//-------------------------------------------------------------------------
#macro Raster(RScale, HLine)
       pigment{ gradient x scale RScale
                color_map{[0.000   color rgbt<1,1,1,0>*0.6]
                          [0+HLine color rgbt<1,1,1,0>*0.6]
                          [0+HLine color rgbt<1,1,1,1>]
                          [1-HLine color rgbt<1,1,1,1>]
                          [1-HLine color rgbt<1,1,1,0>*0.6]
                          [1.000   color rgbt<1,1,1,0>*0.6]} }
 #end// of Raster(RScale, HLine)-macro
//-------------------------------------------------------------------------


plane { <0,1,0>, 0    // plane with layered textures
        texture { pigment{color White*1.1}
                  finish {ambient 0.45 diffuse 0.85}}
        texture { Raster(RasterScale,RasterHalfLine ) rotate<0,0,0> }
        texture { Raster(RasterScale,RasterHalfLineZ) rotate<0,90,0>}
        rotate<0,0,0>
      }
      */
//------------------------------------------------ end of squared plane XZ

//--------------------------------------------------------------------------
//---------------------------- objects in scene ----------------------------
//--------------------------------------------------------------------------
