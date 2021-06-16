#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"

#include <object_slam/Frame.h>
#include <object_slam/g2o_Object.h>
#include <object_slam/matrix_utils.h> 

using namespace cv;
using namespace std;
using namespace Eigen;


Eigen::MatrixXd compute3D_BoxCorner_in_world(Eigen::Vector3d& location, Eigen::Vector3d& dimension, double& ry)
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

void plot_3d_box_with_loc_dim_world(cv::Mat &img, Eigen::MatrixXd& proj_matrix, Eigen::Vector3d&location, Eigen::Vector3d&dimension, double& rot_Y)
{
  Eigen::MatrixXd corner_3d = compute3D_BoxCorner_in_world(location, dimension, rot_Y);
	//   std::cout << "corner_3d: \n" << corner_3d << std::endl;
  Eigen::MatrixXd corner_img = project_world_points_to_2d(corner_3d, proj_matrix);
	//   std::cout << "corner_img: \n" << corner_img << std::endl;
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

int main(int argc, char** argv)
{
    if (argc != 2 )
    {
      cout << "Usage: RGBDPlaneDetection path/to/data" << endl;
      return -1;
    }

    std::string base_folder = std::string(argv[1]);
    std::string pred_objs_txt = base_folder+"online_cubes.txt";  // saved cuboids in local or ground frame.
    std::string init_camera_pose = base_folder+"online_camera.txt"; // offline camera pose for cuboids detection (x y yaw=0, truth roll/pitch/height)
    std::string truth_objs_txt = base_folder+"truth_objects.txt";  // saved ground truth in global frame
    std::string truth_camera_pose = base_folder+"truth_cam_poses.txt";

    Eigen::MatrixXd pred_frame_objects(100,10);  // xyz, roll, pitch, yaw, dimension
    Eigen::MatrixXd init_frame_poses(100,8);	// time/frame, xyz, qwxyz
    Eigen::MatrixXd truth_frame_poses(100,8);	// time/frame, xyz, qwxyz
    Eigen::MatrixXd truth_objects(100,7); // xyz, yaw, dimension // global
    if (!read_all_number_txt(pred_objs_txt,pred_frame_objects))
	return -1;
    if (!read_all_number_txt(init_camera_pose,init_frame_poses))
	return -1;
    if (!read_all_number_txt(truth_camera_pose,truth_frame_poses))
	return -1;
    if (!read_all_number_txt(truth_objs_txt,truth_objects))
	return -1;

    bool whether_plot_ground_truth = false;
    bool whether_plot_offline_cuboid = false;

    Eigen::Matrix3d Kalib; 
    Kalib<<535.4,  0,  320.1,   // for TUM cabinet data.
	    0,  539.2, 247.6,
	    0,      0,     1;  


    int total_frame_number = truth_frame_poses.rows();
    // graph optimization.
    //NOTE in this example, there is only one object!!! perfect association
    g2o::SparseOptimizer graph;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    graph.setAlgorithm(solver);    
	graph.setVerbose(false);
 
    // only first truth pose is used. to directly visually compare with truth pose. also provide good roll/pitch
    // g2o::SE3Quat fixed_init_cam_pose_Twc(truth_frame_poses.row(0).tail<7>());
    g2o::SE3Quat fixed_init_cam_pose_Twc(init_frame_poses.row(0).tail<7>());
    
    // save vertex in every frame
    std::vector<tracking_frame*> all_frames(total_frame_number);    

    int offline_cube_obs_row_id = 0;
    // total_frame_number = 1;
    for (int frame_index = 0; frame_index < total_frame_number; frame_index++)
    {
        // frame_index = 53;
        char frame_index_c[256];
        sprintf(frame_index_c, "%04d", frame_index); // format into 4 digit
        // std::cout << "frame_index: " << frame_index << std::endl;
	    std::cout << "------------frame_index " << frame_index<<"------------" <<std::endl;

        //read rgb image
        // cv::Mat rgb_img = cv::imread(base_folder+"/"+vstrRGBImg[frame_index], CV_LOAD_IMAGE_COLOR);
        cv::Mat rgb_img = cv::imread(base_folder+"/raw_imgs/"+frame_index_c+"_rgb_raw.jpg", CV_LOAD_IMAGE_COLOR);
        if (rgb_img.empty() || rgb_img.depth() != CV_8U)
            cout << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;

        // read truth camera pose
        // only first truth pose is used. to directly visually compare with truth pose. also provide good roll/pitch
        // Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
        Eigen::MatrixXd cam_pose_Twc = init_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
        // std::cout << "cam_pose_Twc: \n" << cam_pose_Twc << std::endl;
        Matrix<double,4,4> transToWolrd;
        transToWolrd.setIdentity();
        transToWolrd.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
        transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
        // std::cout << "transToWolrd: \n" << transToWolrd << std::endl;
        Eigen::Vector3d camera_rpy;
        Eigen::Matrix3d Rot_Mat = transToWolrd.block(0,0,3,3); 
        quat_to_euler_zyx(Quaterniond(Rot_Mat), camera_rpy(0), camera_rpy(1), camera_rpy(2));
        // Eigen::Vector3d camera_rpy = Rot_Mat.eulerAngles(0,1,2);// seems different from quat_to_euler_zyx
        // std::cout << "camera orientation: " << camera_rpy.transpose() << std::endl;

        // // read offline cuboid: each row:  [cuboid_center(3), yaw, cuboid_scale(3), [x1 y1 w h]], prob
        // // std::string pred_frame_obj_txts = base_folder+"/"+vstrCuboid[frame_index];
        // std::string pred_frame_obj_txts = base_folder + "/output/" + frame_index_c + "_3d_cuboids.txt";
        // Eigen::MatrixXd pred_frame_objects(5, 10); // frame_idx, xyz, rpy, dim
        // if (!read_all_number_txt(pred_frame_obj_txts, pred_frame_objects))
        //   return -1;
        Eigen::MatrixXd offline_data = pred_frame_objects.row(frame_index); // xyz, q1234

        if(whether_plot_offline_cuboid)
        {
            Eigen::MatrixXd plot_data(offline_data.rows(), 7); // frame_idx, xyz, yaw, dim
            cv::Mat plot_offline_img = rgb_img.clone();
            for (size_t i = 0; i < offline_data.rows(); i++)
            {
                plot_data(i, 0) = offline_data(i, 1);
                plot_data(i, 1) = offline_data(i, 2);
                plot_data(i, 2) = offline_data(i, 3);
                plot_data(i, 3) = offline_data(i, 6);
                plot_data(i, 4) = offline_data(i, 7);
                plot_data(i, 5) = offline_data(i, 8);
                plot_data(i, 6) = offline_data(i, 9);
                Eigen::Vector3d plot_loc_world = plot_data.row(i).head(3);
                Eigen::Vector3d plot_dim_world = plot_data.row(i).tail(3);
                double plot_yaw = plot_data(i,3);
                Eigen::MatrixXd plot_proj =  Kalib * transToWolrd.inverse().topRows<3>();
                std::cout << "plot_loc_world: " << plot_loc_world.transpose() << std::endl;
                std::cout << "plot_dim_world: " << plot_dim_world.transpose() << std::endl;
                plot_3d_box_with_loc_dim_world(plot_offline_img, plot_proj, plot_loc_world, plot_dim_world, plot_yaw);
                cv::imshow("plot_offline_img", plot_offline_img);
                cv::waitKey(0);
            }// loop object_id
        }// loop plot offline cuboid





        // tracking frame: camera vertex, cuboid vertex every frame, Tcw: optimize result. 
        tracking_frame* currframe = new tracking_frame();
        currframe->frame_seq_id = frame_index;
        all_frames[frame_index] = currframe;
 
        // compute odom val with constant velocity model
        g2o::SE3Quat curr_cam_pose_Twc;
        g2o::SE3Quat odom_val; // from previous frame to current frame
        if (frame_index==0)
            curr_cam_pose_Twc = fixed_init_cam_pose_Twc;
        else
        {
            g2o::SE3Quat prev_pose_Tcw = all_frames[frame_index-1]->cam_pose_Tcw;
            if (frame_index>1)  // from third frame, use constant motion model to initialize camera.
            {
                g2o::SE3Quat prev_prev_pose_Tcw = all_frames[frame_index-2]->cam_pose_Tcw;
                odom_val = prev_pose_Tcw*prev_prev_pose_Tcw.inverse();
            }
            curr_cam_pose_Twc = (odom_val*prev_pose_Tcw).inverse();
        }
        std::cout << "odom_val" << odom_val.toXYZPRYVector().transpose()<< std::endl;

  
        // compute cuboid value with offline txt
        g2o::cuboid cube_local_meas; 
        bool has_detected_cuboid = false;
        has_detected_cuboid = pred_frame_objects(offline_cube_obs_row_id,0)==frame_index;
        if (has_detected_cuboid)  // prepare object measurement   not all frame has observation!!
        {
            // transfer global value to local value as edge measurement
            // Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
            Eigen::MatrixXd cam_pose_Twc = init_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
            Matrix<double,4,4> transToWolrd;
            transToWolrd.setIdentity();
            transToWolrd.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
            transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
            VectorXd measure_data = pred_frame_objects.row(frame_index);
            // std::cout << "measure_data: " << measure_data.transpose() << std::endl;
            Matrix<double,4,4> cube_global_pose; 
            cube_global_pose.setIdentity();
            cube_global_pose.block(0,0,3,3) = euler_zyx_to_rot(measure_data(4), measure_data(5), measure_data(6));
            cube_global_pose.col(3).head(3) = Eigen::Vector3d(measure_data(1), measure_data(2), measure_data(3));
            Matrix<double,4,4>  cube_local_pose = transToWolrd.inverse() * cube_global_pose;
            Eigen::Vector3d obj_rpy;
            Eigen::Matrix3d Rot_Mat = cube_local_pose.block(0,0,3,3); 
            quat_to_euler_zyx(Eigen::Quaterniond(Rot_Mat), obj_rpy(0), obj_rpy(1), obj_rpy(2));
            // std::cout << "obj_rpy: " << obj_rpy.transpose() << std::endl;
            Vector9d cube_pose;
            cube_pose << cube_local_pose.col(3).head(3), obj_rpy, measure_data(7), measure_data(8), measure_data(9);
            cube_local_meas.fromMinimalVector(cube_pose);
            offline_cube_obs_row_id++;
            // print
            g2o::SE3Quat origin_cam_pose_Twc(truth_frame_poses.row(frame_index).tail<7>());
            g2o::cuboid cube_origin_save = cube_local_meas.transform_from(origin_cam_pose_Twc);
            // std::cout << "cube_origin_save: " << cube_origin_save.toMinimalVector().transpose() << std::endl;
            g2o::cuboid cube_before_opti = cube_local_meas.transform_from(curr_cam_pose_Twc);
            // std::cout << "cube_before_opti: " << cube_before_opti.toMinimalVector().transpose() << std::endl;
        } // loop has_detected_cuboid

        if (has_detected_cuboid)
        {
            g2o::VertexCuboid* vCuboid_frame = new g2o::VertexCuboid();
            g2o::cuboid curr_local_value = cube_local_meas;
            vCuboid_frame->setId(0);// if multiple objects, how to set this id
            vCuboid_frame->setEstimate(curr_local_value);
            vCuboid_frame->setFixed(false);
            currframe->cuboids_vertex.push_back(vCuboid_frame);  
        }

        g2o::VertexCuboid* vCube;
        // set up g2o cube vertex. only one in this dataset
        if (frame_index==0)
        {
            g2o::cuboid init_cuboid_global_pose = cube_local_meas.transform_from(curr_cam_pose_Twc);
            vCube = new g2o::VertexCuboid();
            vCube->setId(0);// if multiple objects, how to set this id
            vCube->setEstimate(init_cuboid_global_pose);
            vCube->setFixed(false);
            graph.addVertex(vCube);
        }
        
        // set up g2o camera vertex
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        currframe->pose_vertex = vSE3; // transfer ptr, update after every optimization of every frame
        vSE3->setId(frame_index+1); // +1 because cube is 0
        vSE3->setEstimate(curr_cam_pose_Twc.inverse()); //g2o vertex usually stores world to camera pose.
        vSE3->setFixed(frame_index==0);
        graph.addVertex(vSE3);
        
        // add g2o camera-object measurement edges, if there is
        if (currframe->cuboids_vertex.size()>0)
        {
            g2o::cuboid curr_local_value = currframe->cuboids_vertex[0]->estimate();
            g2o::EdgeSE3Cuboid* e = new g2o::EdgeSE3Cuboid();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vSE3 ));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vCube ));
            e->setMeasurement(curr_local_value);
            e->setId(frame_index);
            Vector9d inv_sigma;inv_sigma<<1,1,1,1,1,1,1,1,1;
            // inv_sigma = inv_sigma*2.0*cube_landmark_meas->meas_quality;
            double meas_quality = 0.75;
            inv_sigma = inv_sigma*2.0*meas_quality;
            Matrix9d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
            e->setInformation(info);
            graph.addEdge(e);
        }
		
     
        // camera vertex, add cam-cam odometry edges
        if (frame_index>0)
        {
            g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( all_frames[frame_index-1]->pose_vertex ));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>( all_frames[frame_index]->pose_vertex ));
            // add to graph, dynamic, read from graph, static
            // e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMObject->mnId + maxKFid + 1))); // object
            e->setMeasurement(odom_val);

            e->setId(total_frame_number+frame_index);
            Vector6d inv_sigma;inv_sigma<<1,1,1,1,1,1;
            inv_sigma = inv_sigma*1.0;
            Matrix6d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
            e->setInformation(info);
            graph.addEdge(e);
        }
        graph.initializeOptimization();
        graph.optimize(5); // do optimization!
        
        // retrieve the optimization result, for debug visualization
        // to get the result, one is vSE3->estimate(), another is vCube->estimate()
        // transfer a ptr to vSE3, also need another ptr for vCube
        for (int j=0;j<=frame_index;j++)
        {
            g2o::VertexSE3Expmap* cam_Tcw_opti=static_cast<g2o::VertexSE3Expmap*>(graph.vertex(j+1));
            all_frames[j]->cam_pose_Tcw = cam_Tcw_opti->estimate();
            // all_frames[j]->cam_pose_Tcw = all_frames[j]->pose_vertex->estimate();
            all_frames[j]->cam_pose_Twc = all_frames[j]->cam_pose_Tcw.inverse();
            std::cout << "cam_pose_Twc: " << all_frames[j]->cam_pose_Twc.toVector().transpose() << std::endl;
        }
        // g2o::cuboid cube_curr = vCube->estimate();
        // std::cout << "cube_final_value: " << cube_curr.toMinimalVector().transpose() << std::endl;
        g2o::VertexCuboid* vCuboid = static_cast<g2o::VertexCuboid*>(graph.vertex(0));
        g2o::cuboid cube_test = vCuboid->estimate();
        std::cout << "cube_final_value: " << cube_test.toMinimalVector().transpose() << std::endl;

	//   if (all_frames[frame_index]->cuboids_vertex.size()>0)
	//   {
    //     g2o::cuboid local_cube = all_frames[frame_index]->cuboids_vertex[0]->estimate();
	      
    //     // print cube after opti, before opti, or just origin
    //     g2o::cuboid global_cube = local_cube.transform_from(all_frames[frame_index]->cam_pose_Twc);
    //     std::cout << "cube_after_opti" << global_cube.toMinimalVector().transpose() << std::endl;
    //     // g2o::cuboid odom_cube = local_cube.transform_from(curr_cam_pose_Twc);
    //     // std::cout << "cube_before_opti" << odom_cube.toMinimalVector().transpose() << std::endl;
    //     // g2o::SE3Quat origin_cam_pose_Twc(truth_frame_poses.row(frame_index).tail<7>());// time x y z qx qy qz qw
    //     // g2o::cuboid origin_cube = local_cube.transform_from(origin_cam_pose_Twc);
    //     // std::cout << "cube_origin_opti" << origin_cube.toMinimalVector().transpose() << std::endl;		
	//   }

    } // frame idx

    cout<<"Finish all optimization! please change to ros for visualization."<<endl;

} // main