#pragma once

#include <string>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>

#include "detect_cuboid_bbox/detect_cuboid_bbox.h"

using namespace Eigen;



// check if point lies inside 2d rect
bool check_inside_box(const Vector2d &pt, const Vector2d &box_left_top, const Vector2d &box_right_bottom);
// (-pi,pi]
void atan2_vector(const VectorXd &y_vec, const VectorXd &x_vec, VectorXd &all_angles);

// compute two line intersection points, a simplified version compared to matlab
Vector2d lineSegmentIntersect(const Vector2d &pt1_start, const Vector2d &pt1_end, const Vector2d &pt2_start, const Vector2d &pt2_end,
							  bool infinite_line = true);

// merge short edges into long. edges n*4  each edge should start from left to right!
void merge_break_lines(const MatrixXd &all_lines, MatrixXd &merge_lines_out, double pre_merge_dist_thre, double pre_merge_angle_thre_degree,
					   double edge_length_threshold = -1);

// calculate_line_length_and_angle
void calculate_line_length_and_angle(Eigen::MatrixXd& line_all, Eigen::MatrixXd& line_angle);

void sort_line_by_index(Eigen::MatrixXd& line_angle, size_t& angle_idx);
void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void filter_line_by_coincide(Eigen::MatrixXd& line_in_order, double& delta_angle, double& delta_distance);


// plot: 
// each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
void plot_image_with_edges(const cv::Mat &rgb_img, cv::Mat &output_img, MatrixXd &all_lines, const cv::Scalar &color);
MatrixXd compute3D_BoxCorner_in_camera(Eigen::Vector3d& dimension, Eigen::Vector3d& location, Eigen::Matrix3d& local_rot_mat);
Eigen::MatrixXd project_camera_points_to_2d(Eigen::MatrixXd& points3d_3x8, Matrix3d& Kalib);
void plot_3d_box_with_loc_dim_camera(cv::Mat &img, Eigen::Matrix3d& Kalib, Eigen::Vector3d& location, Eigen::Vector3d& dimension, Eigen::Matrix3d& local_rot_mat);

// calculate errors: line distance error angle error and obj-plane error
double box_edge_sum_dists(const cv::Mat &dist_map, const MatrixXd &box_corners_2d, const MatrixXi &edge_pt_ids, bool reweight_edge_distance);
double box_edge_angle_error(const MatrixXd &det_lines, const MatrixXd &box_visual_edges);
double box_edge_angle_error_new(const MatrixXd &det_lines, const MatrixXd &box_visual_edges);
double compute_obj_plane_error(Eigen::Vector3d& location, Eigen::Vector3d& dimension, 
                    Eigen::Matrix3d& local_rot_mat, std::vector<cv::Mat>& mvPlaneCoefficients);


// calculate translation
void trasform_rotation_from_world_to_camera(Eigen::Matrix4d& transToWolrd, double& yaw, Eigen::Matrix3d& rot_matrix);
void calculate_location_new(Eigen::Vector3d& dimension, Eigen::MatrixXd& cam_to_img, Eigen::Vector4d& bbox, Matrix3d& local_rot_mat, double& theta_ray, Eigen::Vector3d& location);
void calculate_theta_ray(cv::Mat& img, Eigen::Vector4d& box_2d, Eigen::MatrixXd& proj_matrix, double& theta_ray);
