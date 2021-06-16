#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "object_slam/g2o_Object.h"

// class object_landmark;

class tracking_frame{
public:  

    int frame_seq_id;    // image topic sequence id, fixed
    cv::Mat frame_img;
    cv::Mat cuboids_2d_img;
    
    g2o::VertexSE3Expmap* pose_vertex; // camera origin pose, add to graph
    
    // std::vector<object_landmark*> observed_cuboids; // generated cuboid from this frame. maynot be actual SLAM landmark
    std::vector<g2o::VertexCuboid*> cuboids_vertex; // cuboid origin pose, local cuboid value, maynot be actual SLAM landmark
    

    std::vector<g2o::cuboid> cuboid_pose_origin; // cuboid origin pose, global value, just save
    std::vector<g2o::cuboid> cuboid_pose_local; // cuboid local cuboid value, set to graph edge
    std::vector<int> cuboids_name_id;           // cuboid class name id
    g2o::SE3Quat cam_pose_origin_Tcw;	     // camera origin pose  world to cam
    g2o::SE3Quat cam_odom_origin;	     // camera odom, set to graph edge  world to cam

    g2o::SE3Quat cam_pose_Tcw;	     // optimized pose  world to cam
    g2o::SE3Quat cam_pose_Twc;	     // optimized pose  cam to world
    g2o::cuboid cuboid_pose_opti;   // optimized pose  global value
};