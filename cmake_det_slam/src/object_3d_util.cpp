#include "detect_cuboid_bbox/object_3d_util.h"
#include "detect_cuboid_bbox/matrix_utils.h"

#include <iostream>
#include <numeric>
// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace Eigen;
using namespace std;

// print cuboid info
void cuboid::print_cuboid()
{
    std::cout << "printing cuboids info...." << std::endl;
    std::cout << "pos   " << pos.transpose() << std::endl;
    std::cout << "scale   " << scale.transpose() << std::endl;
    std::cout << "rotY   " << rotY << std::endl;
    // std::cout << "box_config_type   " << box_config_type.transpose() << std::endl;
    // std::cout << "box_corners_2d \n"
    //           << box_corners_2d << std::endl;
    // std::cout << "box_corners_3d_world \n"
    //           << box_corners_3d_world << std::endl;
}

// check if point lies inside 2d rect
bool check_inside_box(const Vector2d &pt, const Vector2d &box_left_top, const Vector2d &box_right_bottom)
{
    return box_left_top(0) <= pt(0) && pt(0) <= box_right_bottom(0) && box_left_top(1) <= pt(1) && pt(1) <= box_right_bottom(1);
}

void atan2_vector(const VectorXd &y_vec, const VectorXd &x_vec, VectorXd &all_angles)
{
    all_angles.resize(y_vec.rows());
    for (int i = 0; i < y_vec.rows(); i++)
        all_angles(i) = std::atan2(y_vec(i), x_vec(i)); // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
}

// compute two line intersection points, a simplified version compared to matlab
Vector2d lineSegmentIntersect(const Vector2d &pt1_start, const Vector2d &pt1_end, const Vector2d &pt2_start, const Vector2d &pt2_end,
                              bool infinite_line)
{
    // treat as [x1 y1 x2 y2]    [x3 y3 x4 y4]
    double X2_X1 = pt1_end(0) - pt1_start(0);
    double Y2_Y1 = pt1_end(1) - pt1_start(1);
    double X4_X3 = pt2_end(0) - pt2_start(0);
    double Y4_Y3 = pt2_end(1) - pt2_start(1);
    double X1_X3 = pt1_start(0) - pt2_start(0);
    double Y1_Y3 = pt1_start(1) - pt2_start(1);
    double u_a = (X4_X3 * Y1_Y3 - Y4_Y3 * X1_X3) / (Y4_Y3 * X2_X1 - X4_X3 * Y2_Y1);
    double u_b = (X2_X1 * Y1_Y3 - Y2_Y1 * X1_X3) / (Y4_Y3 * X2_X1 - X4_X3 * Y2_Y1);
    double INT_X = pt1_start(0) + X2_X1 * u_a;
    double INT_Y = pt1_start(1) + Y2_Y1 * u_a;
    double INT_B = double((u_a >= 0) && (u_a <= 1) && (u_b >= 0) && (u_b <= 1));
    if (infinite_line)
        INT_B = 1;

    return Vector2d(INT_X * INT_B, INT_Y * INT_B);
}

// merge short edges into long. edges n*4  each edge should start from left to right!
void merge_break_lines(const MatrixXd &all_lines, MatrixXd &merge_lines_out, double pre_merge_dist_thre,
                       double pre_merge_angle_thre_degree, double edge_length_threshold)
{
    bool can_force_merge = true;
    merge_lines_out = all_lines;
    int total_line_number = merge_lines_out.rows(); // line_number will become smaller and smaller, merge_lines_out doesn't change
    int counter = 0;
    double pre_merge_angle_thre = pre_merge_angle_thre_degree / 180.0 * M_PI;
    while ((can_force_merge) && (counter < 500))
    {
        counter++;
        can_force_merge = false;
        MatrixXd line_vector = merge_lines_out.topRightCorner(total_line_number, 2) - merge_lines_out.topLeftCorner(total_line_number, 2);
        VectorXd all_angles;
        atan2_vector(line_vector.col(1), line_vector.col(0), all_angles); // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
        for (int seg1 = 0; seg1 < total_line_number - 1; seg1++)
        {
            for (int seg2 = seg1 + 1; seg2 < total_line_number; seg2++)
            {
                double diff = std::abs(all_angles(seg1) - all_angles(seg2));
                double angle_diff = std::min(diff, M_PI - diff);
                if (angle_diff < pre_merge_angle_thre)
                {
                    double dist_1ed_to_2 = (merge_lines_out.row(seg1).tail(2) - merge_lines_out.row(seg2).head(2)).norm();
                    double dist_2ed_to_1 = (merge_lines_out.row(seg2).tail(2) - merge_lines_out.row(seg1).head(2)).norm();

                    if ((dist_1ed_to_2 < pre_merge_dist_thre) || (dist_2ed_to_1 < pre_merge_dist_thre))
                    {
                        Vector2d merge_start, merge_end;
                        if (merge_lines_out(seg1, 0) < merge_lines_out(seg2, 0))
                            merge_start = merge_lines_out.row(seg1).head(2);
                        else
                            merge_start = merge_lines_out.row(seg2).head(2);

                        if (merge_lines_out(seg1, 2) > merge_lines_out(seg2, 2))
                            merge_end = merge_lines_out.row(seg1).tail(2);
                        else
                            merge_end = merge_lines_out.row(seg2).tail(2);

                        double merged_angle = std::atan2(merge_end(1) - merge_start(1), merge_end(0) - merge_start(0));
                        double temp = std::abs(all_angles(seg1) - merged_angle);
                        double merge_angle_diff = std::min(temp, M_PI - temp);
                        if (merge_angle_diff < pre_merge_angle_thre)
                        {
                            merge_lines_out.row(seg1).head(2) = merge_start;
                            merge_lines_out.row(seg1).tail(2) = merge_end;
                            fast_RemoveRow(merge_lines_out, seg2, total_line_number); //also decrease  total_line_number
                            can_force_merge = true;
                            break;
                        }
                    }
                }
            }
            if (can_force_merge)
                break;
        }
    }
    //     std::cout<<"total_line_number after mege  "<<total_line_number<<std::endl;
    if (edge_length_threshold > 0)
    {
        MatrixXd line_vectors = merge_lines_out.topRightCorner(total_line_number, 2) - merge_lines_out.topLeftCorner(total_line_number, 2);
        VectorXd line_lengths = line_vectors.rowwise().norm();
        int long_line_number = 0;
        MatrixXd long_merge_lines(total_line_number, 4);
        for (int i = 0; i < total_line_number; i++)
        {
            if (line_lengths(i) > edge_length_threshold)
            {
                long_merge_lines.row(long_line_number) = merge_lines_out.row(i);
                long_line_number++;
            }
        }
        merge_lines_out = long_merge_lines.topRows(long_line_number);
    }
    else
        merge_lines_out.conservativeResize(total_line_number, NoChange);
}

// calculate_line_length_and_angle
void calculate_line_length_and_angle(Eigen::MatrixXd& line_all, Eigen::MatrixXd& line_angle)
{
  if(line_all.cols()>4)// check length and angle exist
  {
      line_angle.resize(line_all.rows(), line_all.cols());
      line_angle = line_all;
  }
  else // pt1, pt2, length, angle
  {
    // make sure line start from left to right
    for (size_t i = 0; i < line_all.rows(); i++)
    {
      if(line_all(i,0) > line_all(i,2))
      {
          Eigen::Vector2d temp = line_all.row(i).head(2);
          line_all.row(i).head(2) = line_all.row(i).tail(2);
          line_all.row(i).tail(2) = temp;
      }
    }  
    // calculate length and angle
    line_angle.resize(line_all.rows(), 6);
    line_angle.block(0,0,line_all.rows(),4) = line_all;
    for (size_t line_id = 0; line_id < line_all.rows(); line_id++)
    {
        double delta_y = line_all(line_id,3) - line_all(line_id,1);
        double delta_x = line_all(line_id,2) - line_all(line_id,0);
        double length = sqrt(pow(delta_y,2) + pow(delta_x,2));
        double angle = atan2(delta_y, delta_x) /M_PI*180.0;
        if (angle <= 0)
          angle = angle + 180;
        if (angle >= 0 && angle < 10) // make sure 179 and 1 are continuous
          angle = angle + 180;
        line_angle(line_id, 4) = length;
        line_angle(line_id, 5) = angle;
    }
  }
}

void sort_line_by_index(Eigen::MatrixXd& line_angle, size_t& angle_idx)
{
    size_t num = line_angle.rows();
    for (size_t i = 0; i < num; i++)
        for (size_t j = i+1; j < num; j++)
        {
            if (line_angle(i,angle_idx)>line_angle(j,angle_idx))
            {
                Eigen::MatrixXd temp(1,line_angle.cols());
                temp.row(0) = line_angle.row(i);
                line_angle.row(i) = line_angle.row(j);
                line_angle.row(j) = temp.row(0);
            }
        }
    // std::cout << "line_angle: \n" << line_angle << std::endl;
}

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();
    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);
    matrix.conservativeResize(numRows,numCols);
} 
// filter lines
void filter_line_by_coincide(Eigen::MatrixXd& line_in_order, double& delta_angle, double& delta_distance) 
{
    size_t len_idx = 4; // pt1xy, pt2xy, len, angle
    size_t angle_idx = 5;
    Eigen::MatrixXd line_length_angle;
    calculate_line_length_and_angle(line_in_order, line_length_angle);
    sort_line_by_index(line_length_angle, angle_idx);
    Eigen::VectorXi remove_flag(line_length_angle.rows()); // save removed line flag
    remove_flag = Eigen::MatrixXi::Zero(line_length_angle.rows(),1);
    for (int i = 0; i < line_length_angle.rows()-1; i++)
        for (int j = i+1; j < line_length_angle.rows(); j++)
    {
        if(std::abs(line_length_angle(j,angle_idx) - line_length_angle(i,angle_idx)) < delta_angle
          || std::abs(line_length_angle(i,angle_idx) - line_length_angle(j,angle_idx)) < delta_angle) // check delta_angle < 5
        {
            Eigen::Vector2d distance = Eigen::Vector2d(1e5, 1e5);
            Eigen::Vector4d line_1 = line_length_angle.row(j).head<4>();
            Eigen::Vector4d line_2 = line_length_angle.row(i).head<4>();
            for (size_t k = 0; k < 2; k++) // start point, and end point
            {
                Eigen::Vector2d pt = Eigen::Vector2d(line_1(2*k), line_1(2*k+1));
                Eigen::Vector2d line_2_vec = line_2.tail(2)-line_2.head(2);
                // find a point P4, P3P4 and line is vertical, set P4(x,0). 
                // (x4-x3)(x2-x1)+(y4-y3)(y2-y1)=0; >> x4=-(y4-y3)(y2-y1)/(x2-x1)+x3;
                Eigen::Vector2d pt_4 = Eigen::Vector2d(0.0, 0.0);
                pt_4(0) = -(pt_4(1)-pt(1)) * line_2_vec(1) / line_2_vec(0) + pt(0); 
                Eigen::Vector2d intersect = lineSegmentIntersect(pt, pt_4, line_2.head(2), line_2.tail(2), true); 
                Eigen::Vector2d pt_vec = intersect-pt; 
                distance(k) = pt_vec.norm();
            }
            // std::cout << i << " " << j << " distance: " << distance.transpose() << std::endl;
            if(distance.maxCoeff()<delta_distance) // delta_distance=10, check parallel, if so, remove it
            {
                if(line_length_angle(i,len_idx) < line_length_angle(j,len_idx))
                    remove_flag(i) = 1; // we can remove short line, or combine a line, 
                else
                    remove_flag(j) = 1; // we can remove short line, or combine a line, 
                // std::cout << "remove_flag: " << i << " " << remove_flag(i) << " " << j << " " <<remove_flag(j) << std::endl;
            }    
        }
    }
    for (int i = 0; i < line_length_angle.rows(); i++) // remove vertical line
      if(line_length_angle(i, angle_idx)> 80 && line_length_angle(i, angle_idx)< 100)
          remove_flag(i) = 1; 
    
    // std::cout << "remove_flag: " << remove_flag.transpose() << std::endl;
    for (int j = remove_flag.size()-1; j >= 0; j--)
        if(remove_flag(j) == 1)
            removeRow(line_length_angle, j);

    line_in_order.resize(line_length_angle.rows(), 4);
    line_in_order = line_length_angle.block(0,0,line_length_angle.rows(),4);
}



// plot: each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
void plot_image_with_edges(const cv::Mat &rgb_img, cv::Mat &output_img, MatrixXd &all_lines, const cv::Scalar &color)
{
    output_img = rgb_img.clone();
    for (int i = 0; i < all_lines.rows(); i++)
        cv::line(output_img, cv::Point(all_lines(i, 0), all_lines(i, 1)), cv::Point(all_lines(i, 2), all_lines(i, 3)), cv::Scalar(255, 0, 0), 2, 8, 0);
}

MatrixXd compute3D_BoxCorner_in_camera(Eigen::Vector3d& dimension, Eigen::Vector3d& location, Eigen::Matrix3d& local_rot_mat)
{
    MatrixXd corners_body(3, 8);
    corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
                    1, -1, -1, 1, 1, -1, -1, 1,
                    1, 1, 1, 1, -1, -1, -1, -1;
    Matrix3d scale_mat = dimension.asDiagonal();
    Matrix3d rot;
    rot = local_rot_mat;
    // rot << cos(ry), -sin(ry), 0,
    //     sin(ry), cos(ry), 0,
    //     0, 0, 1;                          // rotation around z (up), world coordinate
    // rot << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry); // rotation around y (up), camera coordinate

    MatrixXd corners_without_center = rot * scale_mat * corners_body;
    // std::cout << "dimension\n" << scale_mat * corners_body << std::endl;
    // std::cout << "rot\n" << rot << std::endl;
    // std::cout << "corners_without_center\n" << corners_without_center << std::endl;

    MatrixXd corners_3d(3, 8);
    for (size_t i = 0; i < 8; i++)
    {
      corners_3d(0,i) = corners_without_center(0,i) + location(0);
      corners_3d(1,i) = corners_without_center(1,i) + location(1);
      corners_3d(2,i) = corners_without_center(2,i) + location(2);
    }
    return corners_3d;
}

Eigen::MatrixXd project_camera_points_to_2d(Eigen::MatrixXd& points3d_3x8, Matrix3d& Kalib)
{
  Eigen::MatrixXd corners_2d(3,8);
  corners_2d = Kalib *  points3d_3x8;
  for (size_t i = 0; i < corners_2d.cols(); i++)
  {
      corners_2d(0,i) = corners_2d(0,i) /corners_2d(2,i);
      corners_2d(1,i) = corners_2d(1,i) /corners_2d(2,i);
      corners_2d(2,i) = corners_2d(2,i) /corners_2d(2,i);
  }
  Eigen::MatrixXd corners_2d_return(2,8);
  corners_2d_return = corners_2d.topRows(2);
  return corners_2d_return;
}

void plot_3d_box_with_loc_dim_camera(cv::Mat &img, Eigen::Matrix3d& Kalib, Eigen::Vector3d& location, Eigen::Vector3d& dimension, Eigen::Matrix3d& local_rot_mat)
{
  MatrixXd corner_3d = compute3D_BoxCorner_in_camera(dimension, location, local_rot_mat);
  // MatrixXd corner_3d = compute3D_new(dimension, location, rot_Y);
  // std::cout << "corner_3d: \n" << corner_3d << std::endl;
  MatrixXd corner_img = project_camera_points_to_2d(corner_3d, Kalib);
  // std::cout << "corner_img: \n" << corner_img << std::endl;
  Eigen::MatrixXd edge_pt_ids(2,14); // for 12 body edge (1-2, 2-3, ...) and 2 front line
  edge_pt_ids << 1,2,3,4, 1,2,6,5, 1,4,8,5, 1,2,
                 5,6,7,8, 4,3,7,8, 2,3,7,6, 6,5;
  edge_pt_ids.array()-=1; // transfer 1-8 to 0-7
  for (int pt_id = 0; pt_id < 12; pt_id++) // 12 body edge
  {
    int i = edge_pt_ids(0, pt_id);
    int j = edge_pt_ids(1, pt_id);
    cv::Point pt1 = cv::Point(corner_img(0,i), corner_img(1,i));
    cv::Point pt2 = cv::Point(corner_img(0,j), corner_img(1,j));
    cv::line(img, pt1, pt2, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
  }
  for (int pt_id = 0; pt_id < 2; pt_id++) // 2 front edge
  {
    int i = edge_pt_ids(0, 12+pt_id);
    int j = edge_pt_ids(1, 12+pt_id);
    cv::Point pt1 = cv::Point(corner_img(0,i), corner_img(1,i));
    cv::Point pt2 = cv::Point(corner_img(0,j), corner_img(1,j));
    cv::line(img, pt1, pt2, cv::Scalar(255, 0, 0), 1, CV_AA, 0);
  }

  for (int i = 0; i < 8; i++) // plot 8 corners
  {
    cv::Point pt = cv::Point(corner_img(0,i), corner_img(1,i));
    cv::circle(img, pt, i, CV_RGB(255, 0, 0), 2);
  }

}


// calculate errors: line distance error
double box_edge_sum_dists(const cv::Mat &dist_map, const MatrixXd &box_corners_2d, const MatrixXi &edge_pt_ids, bool reweight_edge_distance)
{
    // give some edges, sample some points on line then sum up distance from dist_map
    // input: visible_edge_pt_ids is n*2  each row stores an edge's two end point's index from box_corners_2d
    // if weight_configs: for configuration 1, there are more visible edges compared to configuration2, so we need to re-weight
    // [1 2;2 3;3 4;4 1;2 6;3 5;4 8;5 8;5 6];  reweight vertical edge id 5-7 by 2/3, horizontal edge id 8-9 by 1/2
    float sum_dist = 0;
    for (int edge_id = 0; edge_id < edge_pt_ids.rows(); edge_id++)
    {
        Vector2d corner_tmp1 = box_corners_2d.col(edge_pt_ids(edge_id, 0));
        Vector2d corner_tmp2 = box_corners_2d.col(edge_pt_ids(edge_id, 1));
        for (double sample_ind = 0; sample_ind < 11; sample_ind++)
        {
            Vector2d sample_pt = sample_ind / 10.0 * corner_tmp1 + (1 - sample_ind / 10.0) * corner_tmp2;
            float dist1 = dist_map.at<float>(int(sample_pt(1)), int(sample_pt(0))); //make sure dist_map is float type
            if (reweight_edge_distance)
            {
                if ((4 <= edge_id) && (edge_id <= 5))
                    dist1 = dist1 * 3.0 / 2.0;
                if (6 == edge_id)
                    dist1 = dist1 * 2.0;
            }
            sum_dist = sum_dist + dist1;
        }
    }
    return double(sum_dist);
}
// calculate errors: line angle error
double box_edge_angle_error(const MatrixXd &det_lines, const MatrixXd &box_visual_edges)
{
  VectorXd det_lines_angles(det_lines.rows());
  MatrixXd det_lines_mid_pt(det_lines.rows(), 2);
  for (int i = 0; i < det_lines.rows(); i++)
  {
    det_lines_angles(i) = std::atan2(det_lines(i, 3) - det_lines(i, 1), det_lines(i, 2) - det_lines(i, 0)); // [-pi -pi]
    if(det_lines_angles(i) < 0)
      det_lines_angles(i) = det_lines_angles(i) + M_PI; // [0,pi]
    det_lines_mid_pt.row(i).head<2>() = (det_lines.row(i).head<2>() + det_lines.row(i).tail<2>()) / 2;
  }
  // std::cout << "det_lines_angles: " << det_lines_angles.transpose() << std::endl;
  // std::cout << "det_lines_angles: " << det_lines_angles.transpose()*180/M_PI << std::endl;
  VectorXd box_edges_angles(box_visual_edges.rows());
  MatrixXd box_edges_mid_pt(box_visual_edges.rows(), 2);
  for (size_t i = 0; i < box_visual_edges.rows(); i++)
  {
    box_edges_angles(i) = std::atan2(box_visual_edges(i, 3) - box_visual_edges(i, 1), box_visual_edges(i, 2) - box_visual_edges(i, 0)); // [-pi -pi]
    if(box_edges_angles(i) < 0)
      box_edges_angles(i) = box_edges_angles(i) + M_PI; // [0,pi]
    box_edges_mid_pt.row(i).head<2>() = (box_visual_edges.row(i).head<2>() + box_visual_edges.row(i).tail<2>()) / 2;
  }
  // std::cout << "box_edges_angles: " << box_edges_angles.transpose() << std::endl;
  // std::cout << "box_edges_angles: " << box_edges_angles.transpose()*180/M_PI << std::endl;

  // for every detected line, find match lines and angle error, if not find, give max
  Eigen::VectorXd det_line_delta_angle(det_lines.rows());
  for (size_t i = 0; i < det_lines.rows(); i++)
  {
    det_line_delta_angle(i) = 30.0 / 180.0 * M_PI ; // if not found any VP supported lines, give each box edge a constant cost (45 or 30 ? degree)
    double angle_thre = 10 / 180.0 * M_PI;
    double min_distance = 500;
    for (size_t j = 0; j < box_visual_edges.rows(); j++)
    {
      double delta_angle = std::abs(det_lines_angles(i) - box_edges_angles(j));
      delta_angle = std::min(delta_angle, M_PI-delta_angle);
      double delta_dis = sqrt(pow(det_lines_mid_pt(i,0)-box_edges_mid_pt(j,0),2)
                              + pow(det_lines_mid_pt(i,1)-box_edges_mid_pt(j,1),2));
      // std::cout << i << " delta_angle: " << delta_angle*180/M_PI << " delta_dis: " << delta_dis << std::endl;

      if(delta_angle < angle_thre && delta_dis < min_distance)
      {
        det_line_delta_angle(i) = delta_angle;
        min_distance = delta_dis;
      }

    }
  }
  // std::cout << "det_lines: \n" << det_lines << std::endl;
  // std::cout << "det_line_delta_angle: " << det_line_delta_angle.transpose()*180/M_PI << std::endl;
  for (size_t i = 0; i < det_lines.rows(); i++)
  {
    if(det_lines_angles(i) > 0 && det_lines_angles(i) < 30.0 / 180.0 * M_PI) // for horizal line, add more weights
      det_line_delta_angle(i) = det_line_delta_angle(i)*2;
    else if (det_lines_angles(i) > 150.0 / 180.0 * M_PI && det_lines_angles(i) <  M_PI)
      det_line_delta_angle(i) = det_line_delta_angle(i)*2;
  }
  return det_line_delta_angle.sum();
}

double box_edge_angle_error_new(const MatrixXd &det_lines, const MatrixXd &box_visual_edges)
{
  VectorXd det_lines_angles(det_lines.rows());
  for (int i = 0; i < det_lines.rows(); i++)
  {
    det_lines_angles(i) = std::atan2(det_lines(i, 3) - det_lines(i, 1), det_lines(i, 2) - det_lines(i, 0)); // [-pi -pi]
    if(det_lines_angles(i) < 0)
      det_lines_angles(i) = det_lines_angles(i) + M_PI; // [0,pi]
  }
  // std::cout << "det_lines_angles: " << det_lines_angles.transpose() << std::endl;
  std::cout << "det_lines_angles: " << det_lines_angles.transpose()*180/M_PI << std::endl;
  VectorXd box_edges_angles(box_visual_edges.rows());
  for (size_t i = 0; i < box_visual_edges.rows(); i++)
  {
    box_edges_angles(i) = std::atan2(box_visual_edges(i, 3) - box_visual_edges(i, 1), box_visual_edges(i, 2) - box_visual_edges(i, 0)); // [-pi -pi]
    if(box_edges_angles(i) < 0)
      box_edges_angles(i) = box_edges_angles(i) + M_PI; // [0,pi]
  }
  // std::cout << "box_edges_angles: " << box_edges_angles.transpose() << std::endl;
  // std::cout << "box_edges_angles: " << box_edges_angles.transpose()*180/M_PI << std::endl;

  // for every detected line, find match lines and angle error, if not find, give max
  Eigen::VectorXd det_line_delta_angle(det_lines.rows());
  for (size_t i = 0; i < det_lines.rows(); i++)
  {
    det_line_delta_angle(i) = 30.0 / 180.0 * M_PI ; // if not found any VP supported lines, give each box edge a constant cost (45 or 30 ? degree)
    double min_angle_delta = 10 / 180.0 * M_PI;
    double min_distance = 500;
    for (size_t j = 0; j < box_visual_edges.rows(); j++)
    {
      double delta_angle = std::abs(det_lines_angles(i) - box_edges_angles(j));
      delta_angle = std::min(delta_angle, M_PI-delta_angle);
      Eigen::Vector2d distance = Eigen::Vector2d(1e5, 1e5);
      Eigen::Vector4d line_1 = det_lines.row(j).head<4>();
      Eigen::Vector4d line_2 = box_visual_edges.row(i).head<4>();
      for (size_t k = 0; k < 2; k++) // start point, and end point
      {
          Eigen::Vector2d pt = Eigen::Vector2d(line_1(2*k), line_1(2*k+1));
          Eigen::Vector2d line_2_vec = line_2.tail(2)-line_2.head(2);
          // find a point P4, P3P4 and line is vertical, set P4(x,0). 
          // (x4-x3)(x2-x1)+(y4-y3)(y2-y1)=0; >> x4=-(y4-y3)(y2-y1)/(x2-x1)+x3;
          Eigen::Vector2d pt_4 = Eigen::Vector2d(0.0, 0.0);
          pt_4(0) = -(pt_4(1)-pt(1)) * line_2_vec(1) / line_2_vec(0) + pt(0); 
          Eigen::Vector2d intersect = lineSegmentIntersect(pt, pt_4, line_2.head(2), line_2.tail(2), true); 
          Eigen::Vector2d pt_vec = intersect-pt; 
          distance(k) = pt_vec.norm();
      }
      std::cout << i << " " << j << " angle_delta " << delta_angle << " distance: " << distance.transpose() << std::endl;
      if(delta_angle<min_angle_delta && distance.maxCoeff()<min_distance)
      {
        det_line_delta_angle(i) = delta_angle;
        min_angle_delta = delta_angle;
        // min_distance = distance.maxCoeff();
      }   

    }
  }
  // std::cout << "det_lines: \n" << det_lines << std::endl;
  std::cout << "det_line_delta_angle: " << det_line_delta_angle.transpose()*180/M_PI << std::endl;
  // for (size_t i = 0; i < det_lines.rows(); i++)
  // {
  //   if(det_lines_angles(i) > 0 && det_lines_angles(i) < 30) // for horizal line, add more weights
  //     det_line_delta_angle(i) = det_line_delta_angle(i)*2;
  //   else if (det_lines_angles(i) > 150 && det_lines_angles(i) < 180)
  //     det_line_delta_angle(i) = det_line_delta_angle(i)*2;
  // }
  return det_line_delta_angle.sum();
}
// calculate errors: object plane error
double compute_obj_plane_error(Eigen::Vector3d& location, Eigen::Vector3d& dimension, 
                    Eigen::Matrix3d& local_rot_mat, std::vector<cv::Mat>& mvPlaneCoefficients)
{
    Eigen::Matrix4d cuboid_mat;
    cuboid_mat.setIdentity();
    cuboid_mat.block(0,0,3,3) = local_rot_mat;
    cuboid_mat.col(3).head(3) = location;
    Eigen::MatrixXd cuboid_corners = compute3D_BoxCorner_in_camera(dimension, location, local_rot_mat);
    std::vector<cv::Mat> cuboid_coef;
    for (size_t k = 0; k < 6; k++)
    {
        // n123 = R(theta).transpose()
        float a = cuboid_mat(0, k%3);
        float b = cuboid_mat(1, k%3);
        float c = cuboid_mat(2, k%3);
        float v = cuboid_mat.col(k%3).norm();
        float d = a*cuboid_corners(0,0) + b*cuboid_corners(1,0)+ c*cuboid_corners(2,0);
        if(k >= 3) // use the first or last corner to calculate d
          d = a*cuboid_corners(0,6) + b*cuboid_corners(1,6)+ c*cuboid_corners(2,6);
        cv::Mat coef = (cv::Mat_<float>(4,1) << a/v, b/v, c/v, -d/v);
        if(coef.at<float>(3) < 0)
          coef = -coef;
        // std::cout << "cuboidPlaneCoef: " << coef.t() << std::endl;
        // Eigen::Vector4d coef = Eigen::Vector4d(a/v, b/v, c/v, -d/v);
        // if (coef(3)<0)
        //     coef = -coef;
        cuboid_coef.push_back(coef);
    }

    double dist_error_3d = 0.0;
    double angle_error_3d = 0.0;
    for (size_t xx = 0; xx < mvPlaneCoefficients.size(); xx++)
    {
        // for every plane, there is a dist and angle error
        float single_angle_error = 1.0;
        float single_dist_error = 1.0;

        cv::Mat plane_local = mvPlaneCoefficients[xx];
        // std::cout << "plane_local: " << plane_local.t() << std::endl;
        float min_dist = 0.2; 
        for (size_t yy = 0; yy < cuboid_coef.size(); yy++)
        {
            cv::Mat coef = cuboid_coef[yy];
            float dist = coef.at<float>(3,0) - plane_local.at<float>(3,0);
            float angle = coef.at<float>(0,0) * plane_local.at<float>(0,0) +
                            coef.at<float>(1,0) * plane_local.at<float>(1,0) +
                            coef.at<float>(2,0) * plane_local.at<float>(2,0);
            if((dist < min_dist && dist > -min_dist) && (angle > 0.9397 || angle < -0.9397)) 
            {
              min_dist = abs(dist);
              single_dist_error = abs(dist);
              single_angle_error = 1.0-angle;
            }
        }
        dist_error_3d += single_dist_error;
        angle_error_3d += single_angle_error;
      // std::cout << "single_dist_error: " << single_dist_error <<" single_angle_error " << single_angle_error << std::endl;
    }
    double k1 = 0.1;
    double k2 = 0.4;
    double obj_plane_err = k1*dist_error_3d + k2*angle_error_3d;
    return obj_plane_err;
}

// calculate translation
void calculate_theta_ray(cv::Mat& img, Eigen::Vector4d& box_2d, Eigen::MatrixXd& proj_matrix, double& theta_ray)
{
    double image_width = img.cols;
    double center_x = box_2d(0) + box_2d(2) / 2.0;
    double dx = center_x - image_width / 2.0;
    double focal_length = proj_matrix(0, 0);
    double angle = atan2(fabs(dx), focal_length);
    if(dx < 0)
      angle = -angle;
    theta_ray = angle;
    // std::cout << "image_width: " << image_width << std::endl;
    // std::cout << "box_2d: " << box_2d.transpose()<< std::endl;
    // std::cout << "dx: " << dx << std::endl;
    // std::cout << "theta_ray: " << theta_ray/M_PI*180<< std::endl;
}

void trasform_rotation_from_world_to_camera(Eigen::Matrix4d& transToWolrd, double& yaw, Eigen::Matrix3d& rot_matrix)
{
  Eigen::Matrix3d Twc = transToWolrd.block(0,0,3,3);
  // Twc << 1,0,0, 0,1,0, 0,0,1;
  Eigen::Matrix3d Tcw = Twc.inverse();
  Eigen::Matrix3d rot_global;
  rot_global << cos(yaw), -sin(yaw), 0,
                sin(yaw), cos(yaw), 0,
                0, 0, 1;          // rotation around z (up), world coordinate
  Eigen::Matrix3d rot_local = Tcw*rot_global;
  // std::cout << "rot_global: " << rot_global.eulerAngles(0,1,2).transpose() << std::endl;
  // std::cout << "Tcw: " << Tcw.eulerAngles(0,1,2).transpose() << std::endl;
  // std::cout << "rot_local: " << rot_local.eulerAngles(0,1,2).transpose() << std::endl;
  // std::cout << "rot_local: \n" << rot_local << std::endl;
  rot_matrix = rot_local;
}
// calculate translation
void calculate_location_new(Eigen::Vector3d& dimension, Eigen::MatrixXd& cam_to_img, Eigen::Vector4d& bbox, Matrix3d& local_rot_mat, double& theta_ray, Eigen::Vector3d& location)
// since camera may not be parallel to ground, object in camera coordinate has roll and pitch
// instead of using global angle, we use local rotation matrix (in camera frame)
// the left,right,switch_mult is reset, and constraints are also rearranged
// main idea is to find corresponding corners in different coordinate.
{
    // bbox // xmin, ymin, xmax, ymax
    // std::cout << "bbox: " << bbox.transpose() << std::endl;
    // double orient = alpha + theta_ray;
    // R_Matrix << cos(orient), 0, sin(orient), 0, 1, 0, -sin(orient), 0, cos(orient);

    Eigen::Matrix3d R_Matrix;
    R_Matrix = local_rot_mat;
    Eigen::Vector3d rpy = local_rot_mat.eulerAngles(0,1,2);
    double alpha = rpy(2) - theta_ray;
    // std::cout << "rpy: " << rpy.transpose() << " alpha °: " << alpha*180/M_PI << std::endl;
    if (alpha < -M_PI) // alpha (-180,180)
      alpha = alpha+2*M_PI;

    // camera coodinate, the order is length, width, heigh
    double dx = dimension(0) ;
    double dy = dimension(1) ;
    double dz = dimension(2) ;

    // # below is very much based on trial and error
    // # based on the relative angle, a different configuration occurs
    // # negative is back of car, positive is front
    double left_mult = 1;
    double right_mult = -1;

    // # if the car is facing the oppositeway, switch left and right
    double front_mult = -1;
    double back_mult = 1;
    if (alpha > -0.2) // strang, not 0
    {    front_mult = 1;
        back_mult = -1;
    }
   // # this works but doesnt make much sense
    if (alpha < 90/180.0*M_PI && alpha > -90.0/180.0*M_PI)
    {    left_mult = -1;
        right_mult = 1;
    }
    // # about straight on but opposite way
    if (alpha > 56.0/180.0*M_PI && alpha < 92.0/180.0*M_PI)
    {   left_mult = -1;
        right_mult = -1;
    }
    // // # about straight on and same way
    // if (alpha > -92.0/180.0*M_PI && alpha < -63.0/180.0*M_PI)
    // {    left_mult = 1;
    //     right_mult = 1;
    // }
    if (alpha > -16.0/180.0*M_PI && alpha < -0.3/180.0*M_PI)
    {    front_mult = -1;
        back_mult = -1;
    }
    if (alpha > 166/180.0*M_PI && alpha < 180.0/180.0*M_PI)
    {    front_mult = 1;
        back_mult = 1;
    }

    // std::cout << "left_mult:" << left_mult << std::endl;
    // std::cout << "right_mult:" << right_mult << std::endl;
    // std::cout << "front_mult:" << front_mult << std::endl;
    // std::cout << "back_mult:" << back_mult << std::endl;

    // Eigen::MatrixXd left_constraints(1,3);
    // Eigen::MatrixXd right_constraints(1,3);
    // Eigen::MatrixXd top_constraints(1,3);
    // Eigen::MatrixXd bottom_constraints(1,3);
    // left_constraints.row(0) = Eigen::Vector3d(-dx, +dy, dz).transpose();
    // top_constraints.row(0) = Eigen::Vector3d(dx, dy, dz).transpose();
    // right_constraints.row(0) = Eigen::Vector3d(+dx, -dy, dz).transpose();
    // bottom_constraints.row(0) = Eigen::Vector3d(-dx, -dy, -dz).transpose();
    // Eigen::MatrixXd all_constraints(1,12);
    // all_constraints.block(0,0,1,3) = left_constraints.row(0);
    // all_constraints.block(0,3,1,3) = top_constraints.row(0);
    // all_constraints.block(0,6,1,3) = right_constraints.row(0);
    // all_constraints.block(0,9,1,3) = bottom_constraints.row(0);

    Eigen::MatrixXd left_constraints(2,3);
    Eigen::MatrixXd right_constraints(2,3);
    Eigen::MatrixXd top_constraints(4,3);
    Eigen::MatrixXd bottom_constraints(4,3);
    Eigen::Vector2i negetive_positive = Eigen::Vector2i(-1,1);
    for (size_t i = 0; i < 2; i++)
    {
      int k = negetive_positive(i); // transfrom from (0,1) to (-1, 1)
      // left_constraints.row(i) = Eigen::Vector3d(left_mult*dx, k*dy, -switch_mult*dz).transpose();
      // right_constraints.row(i) = Eigen::Vector3d(right_mult*dx, k*dy, switch_mult*dz).transpose();
      left_constraints.row(i) = Eigen::Vector3d(left_mult*dx, front_mult*dy, k*dz).transpose();
      right_constraints.row(i) = Eigen::Vector3d(right_mult*dx, back_mult*dy, k*dz).transpose();
    }
    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
    {
      int k = negetive_positive(i); // transfrom from (0,1) to (-1, 1)
      int h = negetive_positive(j); // transfrom from (0,1) to (-1, 1)
      // top_constraints.row(i*2+j) = Eigen::Vector3d(k*dx, -dy, h*dz).transpose();
      // bottom_constraints.row(i*2+j) = Eigen::Vector3d(k*dx, dy, h*dz).transpose();
      top_constraints.row(i*2+j) = Eigen::Vector3d(k*dx, h*dy, dz).transpose();
      bottom_constraints.row(i*2+j) = Eigen::Vector3d(k*dx, h*dy, -dz).transpose();
    }
    // std::cout << "left_constraints: \n" << left_constraints<< std::endl;
    // std::cout << "right_constraints: \n" << right_constraints<< std::endl;
    // std::cout << "top_constraints: \n" << top_constraints<< std::endl;
    // std::cout << "bottom_constraints: \n" << bottom_constraints<< std::endl;
    Eigen::MatrixXd all_constraints(64,12);
    size_t cnt_rows=0;
    for (size_t l_id = 0; l_id < 2; l_id++)
      for (size_t r_id = 0; r_id < 2; r_id++)
        for (size_t t_id = 0; t_id < 4; t_id++)
          for (size_t b_id = 0; b_id < 4; b_id++)
          {
            all_constraints.block(cnt_rows,0,1,3) = left_constraints.row(l_id);
            all_constraints.block(cnt_rows,3,1,3) = top_constraints.row(t_id);
            all_constraints.block(cnt_rows,6,1,3) = right_constraints.row(r_id);
            all_constraints.block(cnt_rows,9,1,3) = bottom_constraints.row(b_id);
            cnt_rows++;
          }
    // // how to filter same rows
    // // // True if equal
    // // bool r = a.isApprox(b, 1e-5);
    Eigen::Vector3d best_x;
    double best_error=1e9;
    
    for (size_t constraints_id = 0; constraints_id < all_constraints.rows(); constraints_id++)
    {
      Eigen::MatrixXd A(4,3);
      Eigen::VectorXd b(4);
      for (size_t i = 0; i < 4; i++)
      {
        Eigen::Vector3d constrait = all_constraints.block(constraints_id,i*3,1,3).transpose(); // left,right,top,or bottom
        Eigen::Vector3d RX_vec = R_Matrix * constrait;
        // Eigen::Vector3d RX_vec_temp = R_Matrix * constrait;
        // RX_vec << RX_vec_temp(0), RX_vec_temp(2), RX_vec_temp(1);
        // std::cout << "R_Matrix: \n" << R_Matrix << std::endl;
        // std::cout << "RX_vec: " << RX_vec.transpose() << std::endl;
        Eigen::Matrix4d RX_Matrix;
        RX_Matrix.setIdentity();
        RX_Matrix.col(3).head(3) = RX_vec;

        Eigen::MatrixXd M(3,4);
        M = cam_to_img * RX_Matrix;
        // if (constraints_id == 0)
        //   std::cout << "RX_Matrix: \n" << RX_Matrix<< std::endl;

        if(i%2==0) // i=0,2 left and right, corresponding to xmin,xmax
        {
          A.row(i) = M.block(0,0,1,3) - bbox(i) * M.block(2,0,1,3); // xmin, xmax
          b(i) = bbox(i) * M(2,3) - M(0,3);
        }
        else
        {
          A.row(i) = M.block(1,0,1,3) - bbox(i) * M.block(2,0,1,3); // ymin, ymax
          b(i) = bbox(i) * M(2,3) - M(1,3);
        }
        // std::cout << "The M is:\n" << M << std::endl;
      }

      // // test different solution for Ax=b
      // std::cout << "The A is:\n" << A << std::endl;
      // std::cout << "The b is:\n" << b.transpose() << std::endl;
      // Vector3d x3 = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
      // // double error = (b-A*x).norm();
      // Vector3d x2 = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
      // Vector3d x1 = A.colPivHouseholderQr().solve(b);
      // // Vector3f x2 = A_matrix.llt().solve(b_matrix);
      // // Vector3f x3 = A_matrix.ldlt().solve(b_matrix);
      // std::cout << "The solution is:" << x3.transpose() << " error:" << (b-A*x3).norm() << std::endl;
      // std::cout << "The solution is:" << x1.transpose() << " error:" << (b-A*x1).norm() << std::endl;
      // std::cout << "The solution is:" << x2.transpose() << " error:" << (b-A*x2).norm() << std::endl;
      Vector3d x = (A.transpose()*A).inverse()*A.transpose()*b;
      double error = (b-A*x).norm();
      // std::cout << constraints_id << " solution is:" << x.transpose() << " error:" << (b-A*x).norm() << std::endl;

      // check if all corners are in 2d bbox
      bool included = false;
      Eigen::Matrix3d Kalib = cam_to_img.block(0,0,3,3);
      Eigen::MatrixXd points3d_camera_3x8 = compute3D_BoxCorner_in_camera(dimension, x, local_rot_mat); // same result. 
      Eigen::MatrixXd points2d_camera_2x8 = project_camera_points_to_2d(points3d_camera_3x8, Kalib);
      Eigen::Vector4d bbox_new;
      bbox_new << points2d_camera_2x8.row(0).minCoeff(), points2d_camera_2x8.row(1).minCoeff(),
                  points2d_camera_2x8.row(0).maxCoeff(), points2d_camera_2x8.row(1).maxCoeff();
      Eigen::Vector4d bbox_delta = bbox_new - bbox;
      // std::cout << "bbox_new: " << bbox_new.transpose() << std::endl;
      // std::cout << "bbox_true: " << bbox.transpose() << std::endl;
      // std::cout << "bbox_delta: " << bbox_delta.transpose() << std::endl;
      double delta = 30;
      if( bbox_delta(0) > -delta && bbox_delta(1) > -delta && // xmin, ymin
          bbox_delta(2) < delta && bbox_delta(3) <   delta ) // xmax, ymax
          included = true;

      // included = true;
      if(error < best_error && included)
      {
        best_x = x;
        best_error = error;
        // std::cout << "constrait: " << all_constraints.block(constraints_id,0*3,1,3) << std::endl;
        // std::cout << "contrait: " << all_constraints.block(constraints_id,1*3,1,3) << std::endl;
        // std::cout << "constrait: " << all_constraints.block(constraints_id,2*3,1,3) << std::endl;
        // std::cout << "constrait: " << all_constraints.block(constraints_id,3*3,1,3) << std::endl;
      }
    }

    location = best_x;
    // std::cout << "best solution is:" << location.transpose() << " error:" << best_error << std::endl;
}
