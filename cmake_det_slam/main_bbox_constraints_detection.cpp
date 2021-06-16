// std c
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// third party
#include "tictoc_profiler/profiler.hpp"
#include <line_lbd/line_lbd_allclass.h>
#include <line_lbd/line_descriptor.hpp>

using namespace std;
using namespace Eigen;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

void read_yaml(const string &path_to_yaml, Eigen::Matrix3d & Kalib, float& depth_scale)
{
    // string strSettingPath = path_to_dataset + "/ICL.yaml";
    cv::FileStorage fSettings(path_to_yaml, cv::FileStorage::READ);

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Load camera parameters from settings file

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    Kalib<< fx,  0,  cx,
            0,  fy,  cy,
            0,  0,   1;
		depth_scale = fSettings["DepthMapFactor"];

}

// make sure column size is given. no checks here. row will be adjusted automatically. if more cols given, will be zero.
template <class T>
bool read_all_number_txt(const std::string txt_file_name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &read_number_mat)
{
    if (!std::ifstream(txt_file_name.c_str()))
    {
        std::cout << "ERROR!!! Cannot read txt file " << txt_file_name << std::endl;
        return false;
    }
    std::ifstream filetxt(txt_file_name.c_str());
    int row_counter = 0;
    std::string line;
    if (read_number_mat.rows() == 0)
        read_number_mat.resize(100, 10);

    while (getline(filetxt, line))
    {
        T t;
        if (!line.empty())
        {
            std::stringstream ss(line);
            int colu = 0;
            while (ss >> t)
            {
                read_number_mat(row_counter, colu) = t;
                colu++;
            }
            row_counter++;
            if (row_counter >= read_number_mat.rows()) // if matrix row is not enough, make more space.
                read_number_mat.conservativeResize(read_number_mat.rows() * 2, read_number_mat.cols());
        }
    }
    filetxt.close();

    read_number_mat.conservativeResize(row_counter, read_number_mat.cols()); // cut into actual rows

    return true;
}
template bool read_all_number_txt(const std::string, MatrixXd &);
template bool read_all_number_txt(const std::string, MatrixXi &);

void plot_2d_bbox_with_xywh(cv::Mat&output_img, Eigen::MatrixXd& raw_2d_objs)
{
  // cv::Mat output_img = rgb_img.clone();
  for (size_t box_id = 0; box_id < raw_2d_objs.rows(); box_id++)
  {
    cv::Point pt1 = cv::Point(raw_2d_objs(box_id,0),                      raw_2d_objs(box_id,1));
    cv::Point pt2 = cv::Point(raw_2d_objs(box_id,0),                      raw_2d_objs(box_id,1)+raw_2d_objs(box_id,3));
    cv::Point pt3 = cv::Point(raw_2d_objs(box_id,0)+raw_2d_objs(box_id,2), raw_2d_objs(box_id,1));
    cv::Point pt4 = cv::Point(raw_2d_objs(box_id,0)+raw_2d_objs(box_id,2), raw_2d_objs(box_id,1)+raw_2d_objs(box_id,3));
    cv::line(output_img, pt1, pt2, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
    cv::line(output_img, pt1, pt3, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
    cv::line(output_img, pt4, pt2, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
    cv::line(output_img, pt4, pt3, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
  }
}

void plot_2d_bbox_with_xyxy(cv::Mat&output_img, Eigen::MatrixXd& raw_2d_objs)
{
  // cv::Mat output_img = rgb_img.clone();
  for (size_t box_id = 0; box_id < raw_2d_objs.rows(); box_id++)
  {
    cv::Point pt1 = cv::Point(raw_2d_objs(box_id,0), raw_2d_objs(box_id,1));
    cv::Point pt2 = cv::Point(raw_2d_objs(box_id,0), raw_2d_objs(box_id,3));
    cv::Point pt3 = cv::Point(raw_2d_objs(box_id,2), raw_2d_objs(box_id,1));
    cv::Point pt4 = cv::Point(raw_2d_objs(box_id,2), raw_2d_objs(box_id,3));
    cv::line(output_img, pt1, pt2, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
    cv::line(output_img, pt1, pt3, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
    cv::line(output_img, pt4, pt2, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
    cv::line(output_img, pt4, pt3, cv::Scalar(0, 225, 0), 1, CV_AA, 0);
  }
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

MatrixXd compute3D_new(Eigen::Vector3d& dimension, Eigen::Vector3d& location, double& ry)
{
    MatrixXd corners_body(3, 8);
    corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
                    1, -1, -1, 1, 1, -1, -1, 1,
                    1, 1, 1, 1, -1, -1, -1, -1;
    Matrix3d scale_mat = dimension.asDiagonal();
    Matrix3d rot;
    rot << cos(ry), -sin(ry), 0,
        sin(ry), cos(ry), 0,
        0, 0, 1;                          // rotation around z (front), how to define xyz
    // rot << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry); // rotation around y (up)
    MatrixXd corners_without_center = rot * scale_mat * corners_body;
    MatrixXd corners_3d(3, 8);
    for (size_t i = 0; i < 8; i++)
    {
      corners_3d(0,i) = corners_without_center(0,i) + location(0);
      corners_3d(1,i) = corners_without_center(1,i) + location(1);
      corners_3d(2,i) = corners_without_center(2,i) + location(2);
    }
    return corners_3d;
}

Eigen::MatrixXd project_world_points_to_2d(Eigen::MatrixXd& points3d_3x8, Eigen::MatrixXd& proj_mat_3x4)
{
  Eigen::MatrixXd points3d_4x8(4,8);
  points3d_4x8.block(0,0,3,8) = points3d_3x8;
  for (size_t i = 0; i < points3d_4x8.cols(); i++)
      points3d_4x8(3,i) = 1.0;
  // cout << "points3d_4x8:\n" << points3d_4x8 << endl;

  Eigen::MatrixXd corners_2d(3,8);
  corners_2d = proj_mat_3x4 * points3d_4x8;
  for (size_t i = 0; i < corners_2d.cols(); i++)
  {
      corners_2d(0,i) = corners_2d(0,i) /corners_2d(2,i);
      corners_2d(1,i) = corners_2d(1,i) /corners_2d(2,i);
      corners_2d(2,i) = corners_2d(2,i) /corners_2d(2,i);
  }
  Eigen::MatrixXd corners_2d_return(2,8);
  corners_2d_return = corners_2d.topRows(2);
  // corners_2d_return = corners_2d.topRows(2).cast <int> ();
  // points2d_2x8 = corners_2d.topRows(2);
  // cout << "points2d_2x8:\n" << points2d_2x8 << endl;
  return corners_2d_return;
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

void plot_3d_box_with_loc_dim_world(cv::Mat &img, Eigen::MatrixXd& proj_matrix, Eigen::Vector3d&location, Eigen::Vector3d&dimension, double& rot_Y)
{
  // Matrix3Xd corner_3d = compute3D_BoxCorner(dimension, location, rot_Y);
  MatrixXd corner_3d = compute3D_new(dimension, location, rot_Y);
  // std::cout << "corner_3d: \n" << corner_3d << std::endl;
  MatrixXd corner_img = project_world_points_to_2d(corner_3d, proj_matrix);
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

void plot_image_with_edges(const cv::Mat &rgb_img, cv::Mat &output_img, MatrixXd &all_lines, const cv::Scalar &color)
// each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
{
    output_img = rgb_img.clone();
    for (int i = 0; i < all_lines.rows(); i++)
        cv::line(output_img, cv::Point(all_lines(i, 0), all_lines(i, 1)), cv::Point(all_lines(i, 2), all_lines(i, 3)), cv::Scalar(255, 0, 0), 2, 8, 0);
}

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

void calculate_location(Eigen::Vector3d& dimension, Eigen::MatrixXd& proj_matrix, Eigen::Vector4d& bbox, double& alpha, double& theta_ray, Eigen::Vector3d& location)
{
    double orient = alpha + theta_ray; // global_angle = local + theta_ray
    Eigen::Matrix3d R_Matrix;
    R_Matrix << cos(orient), 0, sin(orient), 0, 1, 0, -sin(orient), 0, cos(orient);
    // std::cout << "alpha: " << alpha/M_PI*180 << std::endl;
    // std::cout << "orient: " << orient/M_PI*180 << std::endl;

    // bbox = bbox + Eigen::Vector4d(60,0,0,0);
    // std::cout << "bbox: " << bbox.transpose() << std::endl;

    // kitti dataset is different, the order is height, width, length
    // double dx = dimension(2) / 2.0;
    // double dy = dimension(0) / 2.0;
    // double dz = dimension(1) / 2.0;
    double dx = dimension(0) ;
    double dy = dimension(1) ;
    double dz = dimension(2) ;

    double left_mult = -1;
    double right_mult = 1;
    double switch_mult = -1;

    // # below is very much based on trial and error
    // # based on the relative angle, a different configuration occurs
    // # negative is back of car, positive is front
    left_mult = 1;
    right_mult = -1;

    // # about straight on but opposite way
    if (alpha < 92.0/180.0*M_PI && alpha > 88.0/180.0*M_PI)
    {   left_mult = 1;
        right_mult = 1;
    }
    // # about straight on and same way
    else if (alpha < -88.0/180.0*M_PI && alpha > -92.0/180.0*M_PI)
    {    left_mult = -1;
        right_mult = -1;
    }
    // # this works but doesnt make much sense
    else if (alpha < 90/180.0*M_PI && alpha > -90.0/180.0*M_PI)
    {    left_mult = -1;
        right_mult = 1;
    }
    // # if the car is facing the oppositeway, switch left and right
    switch_mult = -1;
    if (alpha > 0)
        switch_mult = 1;




    Eigen::MatrixXd left_constraints(2,3);
    Eigen::MatrixXd right_constraints(2,3);
    Eigen::MatrixXd top_constraints(4,3);
    Eigen::MatrixXd bottom_constraints(4,3);
    Eigen::Vector2i negetive_positive = Eigen::Vector2i(-1,1);
    for (size_t i = 0; i < 2; i++)
    {
      int k = negetive_positive(i); // transfrom from (0,1) to (-1, 1)
      left_constraints.row(i) = Eigen::Vector3d(left_mult*dx, k*dy, -switch_mult*dz).transpose();
      right_constraints.row(i) = Eigen::Vector3d(right_mult*dx, k*dy, switch_mult*dz).transpose();
    }

    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
    {
      int k = negetive_positive(i); // transfrom from (0,1) to (-1, 1)
      int h = negetive_positive(j); // transfrom from (0,1) to (-1, 1)
      top_constraints.row(i*2+j) = Eigen::Vector3d(k*dx, -dy, h*dz).transpose();
      bottom_constraints.row(i*2+j) = Eigen::Vector3d(k*dx, dy, h*dz).transpose();
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
    // how to filter same rows
    // // True if equal
    // bool r = a.isApprox(b, 1e-5);
    Eigen::Vector3d best_x;
    double best_error=1e9;
    for (size_t constraints_id = 0; constraints_id < 64; constraints_id++)
    {
      Eigen::MatrixXd A(4,3);
      Eigen::VectorXd b(4);
      for (size_t i = 0; i < 4; i++)
      {
        Eigen::Vector3d constrait = all_constraints.block(constraints_id,i*3,1,3).transpose(); // left,right,top,or bottom
        Eigen::Vector3d RX_vec = R_Matrix * constrait;
        Eigen::Matrix4d RX_Matrix;
        RX_Matrix.setIdentity();
        RX_Matrix.col(3).head(3) = RX_vec;

        Eigen::MatrixXd M(3,4);
        M = proj_matrix * RX_Matrix;

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
      // test different solution for Ax=b
      // std::cout << "The A is:\n" << A << std::endl;
      // std::cout << "The b is:\n" << b.transpose() << std::endl;
      // Vector3d x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
      // double error = (b-A*x).norm();
      // Vector3d x2 = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
      // Vector3d x1 = A.colPivHouseholderQr().solve(b);
      // // Vector3f x2 = A_matrix.llt().solve(b_matrix);
      // // Vector3f x3 = A_matrix.ldlt().solve(b_matrix);
      // std::cout << "The solution is:" << x.transpose() << " error:" << (b-A*x).norm() << std::endl;
      // std::cout << "The solution is:" << x1.transpose() << " error:" << (b-A*x1).norm() << std::endl;
      // std::cout << "The solution is:" << x2.transpose() << " error:" << (b-A*x2).norm() << std::endl;

      Vector3d x = (A.transpose()*A).inverse()*A.transpose()*b;
      double error = (b-A*x).norm();
      // std::cout << "The solution is:" << x.transpose() << " error:" << (b-A*x).norm() << std::endl;
      // location.block(constraints_id,0,1,3) = x.transpose();

      if(error < best_error)
      {
        best_x = x;
        best_error = error;
      }
    }

    location = best_x;
    // location(1) += dimension(1) ; // bring the KITTI center up to the middle of the object
    // std::cout << "The solution is:" << location.transpose() << " error:" << best_error << std::endl;
}

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
      double delta = 20;
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

void fast_RemoveRow(MatrixXd &matrix, int rowToRemove, int &total_line_number)
{
    matrix.row(rowToRemove) = matrix.row(total_line_number - 1);
    total_line_number--;
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
    det_line_delta_angle(i) = 30.0 / 180.0 * M_PI * 2; // if not found any VP supported lines, give each box edge a constant cost (45 or 30 ? degree)
    double angle_thre = 20 / 180.0 * M_PI;
    double min_distance = 500;
    for (size_t j = 0; j < box_visual_edges.rows(); j++)
    {
      double delta_angle = std::abs(det_lines_angles(i) - box_edges_angles(j));
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
  // std::cout << "det_line_delta_angle: " << det_line_delta_angle.transpose()*180/M_PI << std::endl;
  return det_line_delta_angle.sum();
}

// add sample roll, pitch, yaw
template <class T>
void linespace(T starting, T ending, T step, std::vector<T> &res)
{
    res.reserve((ending - starting) / step + 2);
    while (starting <= ending)
    {
        res.push_back(starting);
        starting += step; // TODO could recode to better handle rounding errors
        if (res.size() > 1000)
        {
            std::cout << "Linespace too large size!!!!" << std::endl;
            break;
        }
    }
}
template void linespace(int, int, int, std::vector<int> &);
template void linespace(double, double, double, std::vector<double> &);

template <class T>
Eigen::Matrix<T, 3, 3> euler_zyx_to_rot(const T &roll, const T &pitch, const T &yaw)
{
    T cp = cos(pitch);
    T sp = sin(pitch);
    T sr = sin(roll);
    T cr = cos(roll);
    T sy = sin(yaw);
    T cy = cos(yaw);

    Eigen::Matrix<T, 3, 3> R;
    R << cp * cy, (sr * sp * cy) - (cr * sy), (cr * sp * cy) + (sr * sy),
        cp * sy, (sr * sp * sy) + (cr * cy), (cr * sp * sy) - (sr * cy),
        -sp, sr * cp, cr * cp;
    return R;
}
template Matrix3d euler_zyx_to_rot<double>(const double &, const double &, const double &);
template Matrix3f euler_zyx_to_rot<float>(const float &, const float &, const float &);

template <class T>
void quat_to_euler_zyx(const Eigen::Quaternion<T> &q, T &roll, T &pitch, T &yaw)
{
    T qw = q.w();
    T qx = q.x();
    T qy = q.y();
    T qz = q.z();

    roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
    pitch = asin(2 * (qw * qy - qz * qx));
    yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}
template void quat_to_euler_zyx<double>(const Eigen::Quaterniond &, double &, double &, double &);
template void quat_to_euler_zyx<float>(const Eigen::Quaternionf &, float &, float &, float &);



int main(int argc, char **argv)
{
    string base_folder = argv[1];

    string save_cube_file = base_folder+"online_cubes.txt";
    string save_camera_file = base_folder+"online_camera.txt";
    ofstream online_stream_cube;
    ofstream online_stream_camera;
    online_stream_cube.open(save_cube_file.c_str());
    online_stream_camera.open(save_camera_file.c_str());

    // cuboid truth: -1.52988 0.456476 0.281575  3.10232 0.396318 0.222742 0.272709
    bool whether_plot_detail_images = false;
    bool whether_plot_ground_truth = false;
    bool whether_plot_sample_images = false;
    bool whether_plot_final_scores = true;
    bool whether_sample_camera_rpy = false;
    bool whether_save_cam_obj_data = true;
    bool whether_save_final_images = false;

    // Load camera parameters from settings file
    std::string strSettingPath = base_folder+"/TUM3.yaml";
    Eigen::Matrix3d Kalib;
    float depth_map_scaling;
    read_yaml(strSettingPath, Kalib, depth_map_scaling);

    std::string truth_camera_pose = base_folder+"/truth_cam_poses.txt";// data: time, x, y, z, qx, qy, qz, qw
    Eigen::MatrixXd truth_frame_poses(100,8);
    if (!read_all_number_txt(truth_camera_pose,truth_frame_poses))
	    return -1;
    std::string truth_cuboid_file = base_folder+"/truth_objects.txt";// x, y, z, yaw, longth, width, height
    Eigen::MatrixXd truth_cuboid_list(1,7);
    if (!read_all_number_txt(truth_cuboid_file, truth_cuboid_list))
	    return -1;

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = base_folder+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    vector<string> vstrBboxFilenames;
    string strFile_yolo = base_folder+"/yolov3_bbox.txt";
    LoadImages(strFile_yolo, vstrBboxFilenames, vTimestamps);

    int total_frame_number = truth_frame_poses.rows();
    // total_frame_number = 1;
    for (int frame_index = 0; frame_index < total_frame_number; frame_index++)
    {
      // frame_index = 22; //42;// 250,450;
      char frame_index_c[256];
      sprintf(frame_index_c, "%04d", frame_index); // format into 4 digit
      std::cout << "-----------" << "frame_index: " << frame_index << "-----------" << std::endl;

      //load image
      cv::Mat rgb_img = cv::imread(base_folder+"/"+vstrImageFilenames[frame_index], 1);
      //read cleaned yolo 2d object detection
      Eigen::MatrixXd raw_2d_objs(10,5);  // 2d rect [x1 y1 width height], and prob
      raw_2d_objs.setZero();
      if (!read_all_number_txt(base_folder+"/"+vstrBboxFilenames[frame_index], raw_2d_objs))
      return -1;
      std::cout << "raw_2d_objs: " << raw_2d_objs << std::endl;

      if(!raw_2d_objs.isZero())
      {
        if (whether_plot_ground_truth)
        {
            cv::Mat output_img = rgb_img.clone();
            Eigen::MatrixXd bbox_2d = raw_2d_objs.block(0,0,1,4);
            plot_2d_bbox_with_xywh(output_img,bbox_2d);
            cv::imshow("2d bounding box", output_img);
            cv::waitKey(0);
        }

        // only first truth pose is used. to directly visually compare with truth pose. also provide good roll/pitch
        Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(0).tail<7>(); // xyz, q1234
        // Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
        if(whether_sample_camera_rpy)
          cam_pose_Twc = truth_frame_poses.row(0).tail<7>(); // xyz, q1234
        std::cout << "cam_pose_Twc: \n" << cam_pose_Twc << std::endl;
        Matrix<double,4,4> transToWolrd;
        transToWolrd.setIdentity();
        transToWolrd.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
        transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
        std::cout << "transToWolrd: \n" << transToWolrd << std::endl;
	      Eigen::Vector3d camera_rpy;
        Eigen::Matrix3d Rot_Mat = transToWolrd.block(0,0,3,3); 
        quat_to_euler_zyx(Quaterniond(Rot_Mat), camera_rpy(0), camera_rpy(1), camera_rpy(2));
        // Eigen::Vector3d camera_rpy = Rot_Mat.eulerAngles(0,1,2);// seems different from quat_to_euler_zyx
        std::cout << "camera orientation: " << camera_rpy.transpose() << std::endl;

        std::vector<double> cam_roll_samples;
        std::vector<double> cam_pitch_samples;
        std::vector<double> cam_yaw_samples;
        std::vector<double> obj_yaw_samples;
        // sample camera yaw or object yaw, maybe the same
        linespace<double>(camera_rpy(2) - 90.0 / 180.0 * M_PI, camera_rpy(2) + 90.0 / 180.0 * M_PI, 5.0 / 180.0 * M_PI, cam_yaw_samples);
        // linespace<double>(camera_rpy(2) - 180.0 / 180.0 * M_PI, camera_rpy(2) + 180.0 / 180.0 * M_PI, 5.0 / 180.0 * M_PI, cam_yaw_samples);
        // double yaw_init = camera_rpy(2) - 90.0 / 180.0 * M_PI; // yaw init is directly facing the camera, align with camera optical axis
        // linespace<double>(yaw_init - 90.0 / 180.0 * M_PI, yaw_init + 90.0 / 180.0 * M_PI, 5.0 / 180.0 * M_PI, obj_yaw_samples);
        
        if(whether_sample_camera_rpy)
        {
          // NOTE later if in video, could use previous object yaw..., also reduce search range
          linespace<double>(camera_rpy(0) - 6.0 / 180.0 * M_PI, camera_rpy(0) + 6.0 / 180.0 * M_PI, 3.0 / 180.0 * M_PI, cam_roll_samples);
          linespace<double>(camera_rpy(1) - 6.0 / 180.0 * M_PI, camera_rpy(1) + 6.0 / 180.0 * M_PI, 3.0 / 180.0 * M_PI, cam_pitch_samples);
          // cam_yaw_samples.push_back(camera_rpy(2));
        }
        else
        {
          cam_roll_samples.push_back(camera_rpy(0));
          cam_pitch_samples.push_back(camera_rpy(1));
          // cam_yaw_samples.push_back(camera_rpy(2));
        }

      	Eigen::MatrixXd proj_matrix(3,4);
        proj_matrix = Kalib * transToWolrd.inverse().topRows<3>();
        // proj_matrix = Kalib * transToWolrd.topRows<3>();
        // std::cout << "proj_matrix:\n " << proj_matrix << std::endl;

        // sample camera yaw or object yaw, maybe the same
        // output ground truth for plot
        // double yaw = M_PI - truth_cuboid_list(0,3);
        double yaw = truth_cuboid_list(0,3); // it doesn't matter if yaw = other value
        Eigen::Vector3d location = truth_cuboid_list.block(0,0,1,3).transpose(); // for evaluation
        Eigen::Vector3d dimension = truth_cuboid_list.block(0,4,1,3).transpose(); // can also sample dimension
        std::cout << "yaw ground truth: " << yaw << std::endl;
        std::cout << "dimension ground truth: " << dimension.transpose()  << std::endl;
        std::cout << "location ground truth: " << location.transpose()  << std::endl;
        
        Eigen::Matrix4d object_matrix;
        object_matrix.setIdentity();
        object_matrix.block(0,0,3,3) = euler_zyx_to_rot<double>(0, 0, yaw);
        object_matrix.col(3).head(3) = location;
        Eigen::Matrix4d object_matrix_camera = transToWolrd.inverse()*object_matrix;
        Eigen::Matrix3d obj_rot_camera = object_matrix_camera.block(0,0,3,3);
        Eigen::Vector3d obj_loc_camera = object_matrix_camera.col(3).head(3);
        Eigen::Vector3d obj_dim_camera = Eigen::Vector3d(dimension(0), dimension(2), dimension(1));
        std::cout << "object_matrix: \n" << object_matrix_camera << std::endl;

        if (whether_plot_ground_truth) // when the camera pose is unknown, transtoworld and pojection is unknown, plot is wrong
        {
          cv::Mat plot_img = rgb_img.clone(); // why plot in camera coordinate needs dimension in world coodinate???
          plot_3d_box_with_loc_dim_camera(plot_img, Kalib, obj_loc_camera, dimension, obj_rot_camera);
          // plot_3d_box_with_loc_dim_world(plot_img, proj_matrix, location, dimension, yaw);
          cv::imshow("image ground truth", plot_img);
          cv::waitKey(0);
        }


        // prepare for score, do not need loop every proposal, outside
        // prepare 1, compute canny and distance map for distance error
          // TODO could canny or distance map outside sampling height to speed up!!!!   Then only need to compute canny onces.
          // detect canny edges and compute distance transform  NOTE opencv canny maybe different from matlab. but roughly same
        cv::Mat gray_img;
        if (rgb_img.channels() == 3)
          cv::cvtColor(rgb_img, gray_img, CV_BGR2GRAY);
        else
          gray_img = rgb_img;
        Eigen::Vector4d bbox_canny;
        bbox_canny(0) = std::max(0.0, raw_2d_objs(0)-10.0);
        bbox_canny(1) = std::max(0.0, raw_2d_objs(1)-10.0);
        bbox_canny(2) = std::min(double(rgb_img.cols), raw_2d_objs(2)+raw_2d_objs(0)+10.0);
        bbox_canny(3) = std::min(double(rgb_img.rows), raw_2d_objs(3)+raw_2d_objs(1)+10.0);
        std::cout << "bbox_canny: " << bbox_canny.transpose() << std::endl;
        cv::Rect canny_bbox = cv::Rect(bbox_canny(0), bbox_canny(1), bbox_canny(2)-bbox_canny(0), bbox_canny(3)-bbox_canny(1)); //left, top, width, height
        cv::Mat im_canny;
        cv::Canny(gray_img(canny_bbox), im_canny, 80, 200); // low thre, high thre    im_canny 0 or 255   [80 200  40 100]
        cv::Mat dist_map;
        cv::distanceTransform(255 - im_canny, dist_map, CV_DIST_L2, 3); // dist_map is float datatype
        if (whether_plot_detail_images)
        {
          cv::imshow("im_canny", im_canny);
          cv::Mat dist_map_img;
          cv::normalize(dist_map, dist_map_img, 0.0, 1.0, cv::NORM_MINMAX);
          cv::imshow("normalized distance map", dist_map_img);
          cv::waitKey();
        }

        // prepare 2: compute lbd line for anle error
        //edge detection
        line_lbd_detect line_lbd_obj;
        line_lbd_obj.use_LSD = true;
        line_lbd_obj.line_length_thres = 15;  // remove short edges
        cv::Mat all_lines_mat;
        line_lbd_obj.detect_filter_lines(rgb_img, all_lines_mat);
        Eigen::MatrixXd all_lines_raw(all_lines_mat.rows,4);
        for (int rr=0;rr<all_lines_mat.rows;rr++)
            for (int cc=0;cc<4;cc++)
                all_lines_raw(rr,cc) = all_lines_mat.at<float>(rr,cc);
        // std::cout << "all_lines_raw: " << all_lines_raw << std::endl;
        Eigen::Vector2d bbox_top_left = Eigen::Vector2d(bbox_canny(0), bbox_canny(1));
        Eigen::Vector2d bbox_bot_right = Eigen::Vector2d(bbox_canny(0)+bbox_canny(2), bbox_canny(1)+bbox_canny(3));

        // find edges inside the object bounding box
        MatrixXd all_lines_inside_object(all_lines_raw.rows(), all_lines_raw.cols()); // first allocate a large matrix, then only use the toprows to avoid copy, alloc
        int inside_obj_edge_num = 0;
        for (int edge_id = 0; edge_id < all_lines_raw.rows(); edge_id++)
          if (check_inside_box(all_lines_raw.row(edge_id).head<2>(), bbox_top_left, bbox_bot_right))
            if (check_inside_box(all_lines_raw.row(edge_id).tail<2>(), bbox_top_left, bbox_bot_right))
            {
              all_lines_inside_object.row(inside_obj_edge_num) = all_lines_raw.row(edge_id);
              inside_obj_edge_num++;
            }

        // merge edges and remove short lines, after finding object edges.  edge merge in small regions should be faster than all.
        double pre_merge_dist_thre = 20;
        double pre_merge_angle_thre = 5;
        double edge_length_threshold = 30;
        MatrixXd all_lines_merge_inobj;
        merge_break_lines(all_lines_inside_object.topRows(inside_obj_edge_num), all_lines_merge_inobj, pre_merge_dist_thre,
                  pre_merge_angle_thre, edge_length_threshold);
        // // std::cout << "all_lines_merge_inobj: " << all_lines_merge_inobj << std::endl;
        if (whether_plot_detail_images)
        {
            cv::Mat output_img;
            plot_image_with_edges(rgb_img, output_img, all_lines_merge_inobj, cv::Scalar(255, 0, 0));
            cv::imshow("Raw detected Edges", output_img);
            cv::waitKey(0);
        }


        // sample and find best cuboid
        double combined_scores = 1e9;
        double min_com_error = 1e9;
        double final_yaw_world = 0;
        Matrix3d object_final_rot;
        Eigen::Vector3d final_location_world;
        Eigen::Vector3d final_location_camera;
        Eigen::Vector3d final_camera_vector;

        for (int cam_roll_id = 0; cam_roll_id < cam_roll_samples.size(); cam_roll_id++)
          for (int cam_pitch_id = 0; cam_pitch_id < cam_pitch_samples.size(); cam_pitch_id++)
            for (int cam_yaw_id = 0; cam_yaw_id < cam_yaw_samples.size(); cam_yaw_id++)
            // for (int obj_yaw_id = 0; obj_yaw_id < obj_yaw_samples.size(); obj_yaw_id++)
        {
          // std::cout <<"yaw_id: " << obj_yaw_id << "-----------" << std::endl;
          std::cout <<"yaw_id: " << cam_yaw_id << "-----------" << std::endl;

          //step 1: sample camera rpy or sample object yaw, transfer to camera coordinate 
          // sample object global yaw
          // // double orientation = 0 + i*0.1;// yaw; // sample global yaw
          // double yaw_sample = yaw + yaw_id*0.1;
          // // double yaw_sample = 0 + yaw_id*0.1;
          // std::cout <<"yaw_sample: " << yaw_sample << "-----------" << std::endl;
          // Eigen::Matrix3d obj_local_rot;
          // trasform_rotation_from_world_to_camera(transToWolrd, yaw_sample, obj_local_rot);
          // // std::cout <<"obj_local_rot: \n" << obj_local_rot << std::endl;

          // how about sample camera roll yaw pitch
          Eigen::Vector3d camera_rpy_new;
          camera_rpy_new << cam_roll_samples[cam_roll_id], cam_pitch_samples[cam_pitch_id], cam_yaw_samples[cam_yaw_id];
          // camera_rpy_new << cam_roll_samples[cam_roll_id], cam_pitch_samples[cam_pitch_id], cam_yaw_samples[0];
          std::cout <<"camera_rpy_new: " << camera_rpy_new.transpose() << std::endl;
          Eigen::Matrix3d Rot_Mat_new = euler_zyx_to_rot(camera_rpy_new(0), camera_rpy_new(1), camera_rpy_new(2));
          transToWolrd.block(0,0,3,3) = Rot_Mat_new;
          Eigen::Matrix3d obj_local_rot;
          double yaw_sample = yaw;
          // double yaw_sample = obj_yaw_samples[obj_yaw_id];
          trasform_rotation_from_world_to_camera(transToWolrd, yaw_sample, obj_local_rot);
          // std::cout <<"obj_local_rot: \n" << obj_local_rot << std::endl;
          Eigen::Vector3d object_rpy;
          quat_to_euler_zyx(Quaterniond(obj_local_rot), object_rpy(0), object_rpy(1), object_rpy(2));
          std::cout << "object_rpy: " << object_rpy.transpose() << std::endl;

          // step 2: calcuate object location in camera coordinate
          double theta_ray = 0;
          Eigen::Vector4d box_2d; // change xywh to xyxy
          box_2d << raw_2d_objs(0), raw_2d_objs(1),
                    raw_2d_objs(0)+raw_2d_objs(2), raw_2d_objs(1)+raw_2d_objs(3);
          Eigen::MatrixXd cam_to_img(3,4); // cam_to_img=[K|0]
          cam_to_img.block(0,0,3,3) = Kalib;
          cam_to_img.col(3).head(3) = Eigen::Vector3d(0,0,0);
          calculate_theta_ray(rgb_img, box_2d, cam_to_img, theta_ray);
          // std::cout << "theta_ray in degree: " << theta_ray/M_PI*180 << std::endl;

          Eigen::Vector3d esti_location = Eigen::Vector3d(0,0,0);
          calculate_location_new(dimension, cam_to_img, box_2d, obj_local_rot, theta_ray, esti_location);
          std::cout << "esti location: " << esti_location.transpose() << std::endl;
          
          // plot in camera coordinate
          if(whether_plot_sample_images)
          {
            // if(!esti_location.isZero())
            cv::Mat plot_img = rgb_img.clone();
            plot_3d_box_with_loc_dim_camera(plot_img, Kalib, esti_location, dimension, obj_local_rot);
            cv::imshow("proposal image", plot_img);
            cv::waitKey(0);
          }
          // transfer from camera coordinate to world coodinate
          // Eigen::Vector4d esti_location_4x1;
          // esti_location_4x1 << esti_location, 1;
          // Eigen::Vector3d global_location = transToWolrd.block(0,0,3,4) * esti_location_4x1;
          // std::cout << "esti global_location: " << global_location.transpose() << std::endl;
          // if(whether_plot_detail_images)
          // {
          // cv::Mat plot_img = rgb_img.clone();
          // plot_3d_box_with_loc_dim_world(plot_img, proj_matrix, global_location, dimension, yaw_sample);
          // cv::imshow("proposal image", plot_img);
          // cv::waitKey(0);
          // }


          // step 3: compute visible corners and edge, prepare for score
          // prepare for 2d corner in image
          MatrixXd corner_3d = compute3D_BoxCorner_in_camera(dimension, esti_location, obj_local_rot);
          MatrixXd box_corners_2d_float = project_camera_points_to_2d(corner_3d, Kalib);
          Eigen::Vector4d bbox_new;
          bbox_new << box_corners_2d_float.row(0).minCoeff(), box_corners_2d_float.row(1).minCoeff(),
                      box_corners_2d_float.row(0).maxCoeff(), box_corners_2d_float.row(1).maxCoeff();
          // std::cout << "bbox_new: " << bbox_new.transpose() << std::endl;          

          // prepare for visible 2d corner in image
          // based on trial, may be different from different dataset ...
          Eigen::MatrixXi visible_edge_pt_ids;
          if (object_rpy(2) > 0.3 && object_rpy(2) < 1.33) // nearly 4*90°
          {
            visible_edge_pt_ids.resize(9, 2);
            visible_edge_pt_ids << 1, 2, 2, 3, 3, 4, 4, 1,  1, 5, 2, 6, 3, 7,   5, 6, 6, 7; // 1234 are shown all the time
            visible_edge_pt_ids.array() -= 1;
          }
          else if (object_rpy(2) > -0.21 && object_rpy(2) < 0.3)
          {
            visible_edge_pt_ids.resize(7, 2);
            visible_edge_pt_ids << 1, 2, 2, 3, 3, 4, 4, 1,  2, 6, 3, 7,   6, 7; // 1234 are shown all the time
            visible_edge_pt_ids.array() -= 1;
          }
          else if (object_rpy(2) > -1.21 && object_rpy(2) < -0.21) 
          {
            visible_edge_pt_ids.resize(9, 2);
            visible_edge_pt_ids << 1, 2, 2, 3, 3, 4, 4, 1,  2, 6, 3, 7, 4, 8,   6, 7, 7, 8; 
            visible_edge_pt_ids.array() -= 1;
          }
          else if (object_rpy(2) > -1.81 && object_rpy(2) < -1.21)
          {
            visible_edge_pt_ids.resize(7, 2);
            visible_edge_pt_ids << 1, 2, 2, 3, 3, 4, 4, 1,  3, 7, 4, 8,   7, 8; 
            visible_edge_pt_ids.array() -= 1;
          }
          else if (object_rpy(2) > -2.86 && object_rpy(2) < -1.81) 
          {
            visible_edge_pt_ids.resize(9, 2);
            visible_edge_pt_ids << 1, 2, 2, 3, 3, 4, 4, 1,  3, 7, 4, 8, 1, 5,   7, 8, 5, 8; 
            visible_edge_pt_ids.array() -= 1;
          }
          else if (object_rpy(2) > 2.92 || object_rpy(2) < -2.86) // -3.14=+3.14
          {
            visible_edge_pt_ids.resize(7, 2);
            visible_edge_pt_ids << 1, 2, 2, 3, 3, 4, 4, 1,  4, 8, 1, 5,   5, 8; 
            visible_edge_pt_ids.array() -= 1;
          }
          else if (object_rpy(2) > 1.93 && object_rpy(2) < 2.92) 
          {
            visible_edge_pt_ids.resize(9, 2);
            visible_edge_pt_ids << 1, 2, 2, 3, 3, 4, 4, 1,  4, 8, 1, 5, 2, 6,  5, 8, 5, 6; 
            visible_edge_pt_ids.array() -= 1;
          }
          else if (object_rpy(2) > 1.33 && object_rpy(2) < 1.93) // 
          {
            visible_edge_pt_ids.resize(7, 2);
            visible_edge_pt_ids << 1, 2, 2, 3, 3, 4, 4, 1,  1, 5, 2, 6,   5, 6; 
            visible_edge_pt_ids.array() -= 1;
          }
          // std::cout <<"visible_edge_pt_ids: \n" << visible_edge_pt_ids.transpose() << std::endl;

          // prepare for visible 2d edges in image
          MatrixXd box_edges_visible;
          box_edges_visible.resize(visible_edge_pt_ids.rows(),4);
          for (size_t i = 0; i < box_edges_visible.rows(); i++)
          {
            box_edges_visible(i,0) = box_corners_2d_float(0, visible_edge_pt_ids(i,0));
            box_edges_visible(i,1) = box_corners_2d_float(1, visible_edge_pt_ids(i,0));
            box_edges_visible(i,2) = box_corners_2d_float(0, visible_edge_pt_ids(i,1));
            box_edges_visible(i,3) = box_corners_2d_float(1, visible_edge_pt_ids(i,1));
          }


          // step 4: add score function : ditance error and angle error
          // make sure new bbox are in the canny image, else, distance error is not accurate
          if( bbox_new(0) > bbox_canny(0) && bbox_new(1) > bbox_canny(1) &&   // xmin, ymin
              bbox_new(2) < bbox_canny(2) && bbox_new(3) < bbox_canny(3) ) // xmax, ymax
          {
          // compute distance error
	        bool reweight_edge_distance = false; // if want to compare with all configurations. we need to reweight
          double sum_dist=0.0;
          MatrixXd box_corners_2d_float_shift(2, 8); // shift from whole image to canny image
          box_corners_2d_float_shift.row(0) = box_corners_2d_float.row(0).array() - bbox_canny(0);
          box_corners_2d_float_shift.row(1) = box_corners_2d_float.row(1).array() - bbox_canny(1);
          sum_dist = box_edge_sum_dists(dist_map, box_corners_2d_float_shift, visible_edge_pt_ids, reweight_edge_distance);
			    double obj_diag_length = sqrt(raw_2d_objs(2) * raw_2d_objs(2) + raw_2d_objs(3) * raw_2d_objs(3)); // sqrt(width^2+height^2)
          double distance_error = fabs(sum_dist/obj_diag_length);
          std::cout <<"sum_dist: " << sum_dist << " distance_error: " << distance_error << std::endl;

          // compute angle error
          double angle_error = box_edge_angle_error(all_lines_merge_inobj, box_edges_visible);
          std::cout << "angle_error: " << angle_error << std::endl;

          // compute combined error = (dis_err + k*angle_err)/(1+k)
	        double weight_angle = 2; //0.8 for cabinet, 2.0 for living room
          // double combined_scores = (distance_error + weight_angle * angle_error) / (1 + weight_angle);
          combined_scores = (distance_error + weight_angle * angle_error) / (1 + weight_angle);
          std::cout << "combined_scores: " << combined_scores << std::endl;
          
          }// loop if(in 2d box)
          else
          {
            std::cout << "combined_scores: 10000"  << std::endl;
          } // loop if(in 2d box)

          // step 5: update selected cuboid with the min_error (distance error + angle error)
          if (combined_scores < min_com_error)
          {
            std::cout << "yaw update, combined_scores: " << combined_scores << " min_com_error: "<< min_com_error << std::endl;  
            min_com_error = combined_scores;
            // final_yaw_world = yaw_sample;
            // final_location_world = global_location;
            object_final_rot = obj_local_rot;
            final_location_camera = esti_location;
            final_camera_vector = camera_rpy_new;
          }

        
        } // loop yaw_id

        if(whether_save_cam_obj_data)
        {
          // // save object in camera coordinate, local
          // Eigen::Vector3d object_final_rpy;
          // quat_to_euler_zyx(Quaterniond(object_final_rot), object_final_rpy(0), object_final_rpy(1), object_final_rpy(2));
          // online_stream_cube << frame_index << " " << final_location_camera(0)
          //     << " " << final_location_camera(1)  << " " << final_location_camera(2) 
          //     << " " << object_final_rpy(0) << " " << object_final_rpy(1)
          //     << " " << object_final_rpy(2)
          //     << " " << dimension(0) << " " << dimension(1) << " " << dimension(2)
          //     << " " << "\n";

          // save object in world coordinate (with first frame), ground based
          Eigen::Matrix4d transToWolrd_new;
          transToWolrd_new.setIdentity();
          transToWolrd_new.block(0,0,3,3) = euler_zyx_to_rot(final_camera_vector(0), final_camera_vector(1), final_camera_vector(2));
          transToWolrd_new.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
          Eigen::Matrix4d final_obj_mat;
          final_obj_mat.setIdentity();
          final_obj_mat.block(0,0,3,3) = object_final_rot;
          final_obj_mat.col(3).head(3) = final_location_camera;
          Eigen::Matrix4d final_obj_mat_global = transToWolrd_new * final_obj_mat;
          Eigen::Vector3d object_rpy_global, final_location_global;
          Eigen::Matrix3d mat_temp = final_obj_mat_global.block(0,0,3,3);
          final_location_global = final_obj_mat_global.col(3).head(3);
          quat_to_euler_zyx(Quaterniond(mat_temp), object_rpy_global(0), object_rpy_global(1), object_rpy_global(2));
          std::cout << "esti global_location: " << final_obj_mat_global.col(3).head(3) << std::endl;
          std::cout << "esti global_orient: " << object_rpy_global.transpose() << std::endl;
          online_stream_cube << frame_index << " " << final_location_global(0)
              << " " << final_location_global(1)  << " " << final_location_global(2) 
              << " " << object_rpy_global(0) << " " << object_rpy_global(1)
              << " " << object_rpy_global(2)
              << " " << dimension(0) << " " << dimension(1) << " " << dimension(2)
              << " " << "\n";
          
          Eigen::Matrix3d Rot_Mat_new = euler_zyx_to_rot(final_camera_vector(0), final_camera_vector(1), final_camera_vector(2));
          Eigen::Quaterniond qwxyz_cam = Quaterniond(Rot_Mat_new);
          online_stream_camera << frame_index << " " << cam_pose_Twc(0)
              << " " << cam_pose_Twc(1)  << " " << cam_pose_Twc(2) 
              << " " << qwxyz_cam.x() << " " << qwxyz_cam.y()
              << " " << qwxyz_cam.z() << " " << qwxyz_cam.w()
              << " " << "\n";
        }
        if(whether_plot_final_scores)
        {
          std::cout << "!!!!!!!final_camera_vector:"  << final_camera_vector.transpose()  << std::endl;  
          std::cout << "!!!!!!!final_location_camera:"  << final_location_camera.transpose()  << std::endl;

          cv::Mat plot_img = rgb_img.clone();
          // plot_3d_box_with_loc_dim_world(plot_img, proj_matrix, final_location_world, dimension, final_yaw_world);
          plot_3d_box_with_loc_dim_camera(plot_img, Kalib, final_location_camera, dimension, object_final_rot);
          cv::imshow("selection image", plot_img);
          cv::waitKey(0);
          if (whether_save_final_images)
          {
            std::string save_final_image_with_cuboid = base_folder + "/best_objects_img/" + frame_index_c + "_best_objects.jpg";
            cv::imwrite( save_final_image_with_cuboid, plot_img );
          }
        }


      } // loop if(2d_bbox)
      
      else // else no bbox
      {
        std::cout << "++++++++++++ NO BBOX ++++++++"  << std::endl;
      }
      
    } // loop frame_id

    if(whether_save_cam_obj_data)
    {
        online_stream_cube.close();
        online_stream_camera.close();
    }


    return 0;
}