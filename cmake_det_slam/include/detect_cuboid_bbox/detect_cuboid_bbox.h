#pragma once

// std c
#include <string>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>

#include "detect_cuboid_bbox/matrix_utils.h"

/*
class cuboid // matlab cuboid struct. cuboid on ground. only has yaw, no obj roll/pitch
{
    public:
      Eigen::Vector3d pos;
      Eigen::Vector3d scale;
      double rotY;

      Eigen::Vector2d box_config_type;       // configurations, vp1 left/right
      Eigen::Matrix2Xi box_corners_2d;       // 2*8
      Eigen::Matrix3Xd box_corners_3d_world; // 3*8

      Eigen::Vector4d rect_detect_2d; //% 2D bounding box (might be expanded by me)
      double edge_distance_error;
      double edge_angle_error;
      double normalized_error; // normalized distance+angle
      double skew_ratio;
      double down_expand_height;
      double camera_roll_delta;
      double camera_pitch_delta;

      void print_cuboid(); // print pose information
};
typedef std::vector<cuboid *> ObjectSet; // for each 2D box, the set of generated 3D cuboids

struct cam_pose_infos
{
      Eigen::Matrix4d transToWolrd;
      Eigen::Matrix3d Kalib;

      Eigen::Matrix3d rotationToWorld;
      Eigen::Vector3d euler_angle;
      Eigen::Matrix3d invR;
      Eigen::Matrix3d invK;
      Eigen::Matrix<double, 3, 4> projectionMatrix;
      Eigen::Matrix3d KinvR; // K*invR
      double camera_yaw;
};


class detect_3d_cuboid
{
    public:
      cam_pose_infos cam_pose;
      cam_pose_infos cam_pose_raw;
      void set_calibration(const Eigen::Matrix3d &Kalib);
      void set_cam_pose(const Eigen::Matrix4d &transToWolrd);

      // object detector needs image, camera pose, and 2D bounding boxes(n*5, each row: xywh+prob)  long edges: n*4.  all number start from 0
      void detect_cuboid(const cv::Mat &rgb_img, const Eigen::Matrix4d &transToWolrd, const Eigen::MatrixXd &obj_bbox_coors, Eigen::MatrixXd edges,
                         std::vector<ObjectSet> &all_object_cuboids);
      void detect_cuboid_new(cv::Mat &rgb_img, Eigen::Matrix4d &transToWolrd, const Eigen::MatrixXd &obj_bbox_coors, Eigen::MatrixXd edges,
                         std::vector<ObjectSet> &all_object_cuboids);

      bool whether_plot_detail_images = false;
      bool whether_plot_final_images = false;
      bool whether_save_final_images = false;
      cv::Mat cuboids_2d_img; // save to this opencv mat
      bool print_details = false;

      // important mode parameters for proposal generation.
      bool consider_config_1 = true; // false true
      bool consider_config_2 = true;
      bool whether_sample_cam_roll_pitch = false; // sample camera roll pitch in case don't have good camera pose
      bool whether_sample_bbox_height = false;    // sample object height as raw detection might not be accurate

      int max_cuboid_num = 1;        //final return best N cuboids
      double nominal_skew_ratio = 1; // normally this 1, unless there is priors
      double max_cut_skew = 3;
};

*/


class cuboid
{
public:
    std::string obj_name;
    Eigen::Vector4d bbox_2d; //% 2D bounding box 

    Eigen::Vector3d pos;   // global value
    Eigen::Vector3d scale;    // global value
    double rotY;              // global value

    Eigen::Matrix3d  obj_rot_camera;
    Eigen::Vector3d  obj_loc_camera;
    Eigen::Vector3d  obj_dim_camera;
    double edge_distance_error;
    double edge_angle_error;
    double plane_obj_error;
    double overall_error;

    Eigen::Matrix2Xd box_corners_2d;       // 2*8
    Eigen::Matrix3Xd box_corners_3d_world; // 3*8
    Eigen::Matrix3Xd box_corners_3d_cam; // 3*8

    void print_cuboid(); // print pose information
};
typedef std::vector<cuboid *> ObjectSet; // for each 2D box, the set of generated 3D cuboids

class detect_cuboid_bbox
{
public:
    bool whether_plot_detail_images = false;
    bool whether_plot_ground_truth = false;
    bool whether_plot_sample_images = false;
    bool whether_plot_final_scores = false;
    bool whether_sample_obj_dimension = true;
    bool whether_sample_obj_yaw = true;
    bool whether_add_plane_constraints = true;
    bool whether_save_cam_obj_data = true;
    bool whether_save_final_image = true;

    cv::Mat rgb_img;
    // Eigen::Matrix4d Twc; // transToWolrd
    Eigen::Matrix3d Kalib;
    double mDepthMapFactor;

	std::vector<std::string> detected_obj_name;
    Eigen::MatrixXd det_bbox_2d;
    Eigen::MatrixXd truth_cuboid_list;
    Eigen::MatrixXd truth_frame_poses;

public:
    bool Read_Image_TUM(std::string & img_file); 
    bool Read_Kalib_TUM(std::string &calib_file);
    bool Read_Bbox_2D_TUM(std::string & bbox_file);
    bool Read_Camera_Pose_TUM(std::string & cam_pose_file);
    bool Read_Object_Info_TUM(std::string & cuboid_file);

    void detect_cuboid_every_frame(cv::Mat& rgb_img, std::vector<cv::Mat>& mvPlaneCoefficients, Eigen::Matrix4d& Twc,
		std::vector<ObjectSet>& frame_cuboid, std::string& output_file, std::string& output_img);
    void detect_cuboid_with_bbox_constraints(cv::Mat& rgb_img, Eigen::Matrix4d& Twc, Eigen::Vector4d& obj_bbox, double& obj_yaw, 
            const Eigen::Vector3d& obj_dim_ave, std::vector<cv::Mat>& mvPlaneCoefficients, std::vector<cuboid *>& single_object_candidate);
    void compute_obj_visible_edge(cuboid* new_cuboid, 
        Eigen::MatrixXi& visible_edge_pt_ids, Eigen::MatrixXd& box_edges_visible);
    void formulate_cuboid_param(cuboid* new_cuboid, Eigen::Vector3d& obj_loc_cam, 
            Eigen::Matrix3d& obj_rot_cam, Eigen::Vector3d& obj_dim_cam, Eigen::Matrix3d& Kalib);
};