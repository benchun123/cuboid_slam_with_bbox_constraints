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

// ours
#include "dataset_tum.h"
#include "detect_cuboid_bbox/detect_cuboid_bbox.h"
#include "plane_detection.h"

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: det_bbox_tum_node path/to/data" << endl;
        return -1;
    }

    ca::Profiler::enable();
    std::string base_folder = std::string(argv[1]);
    std::string calib_file = base_folder + "/TUM3.yaml";
    std::string rgb_file = base_folder + "/rgb.txt";
    std::string depth_file = base_folder + "/depth.txt";
    std::string bbox_2d_list_file = base_folder+"/yolov3_bbox.txt";
    std::string truth_camera_pose = base_folder+"/truth_cam_poses.txt";// data: time, x, y, z, qx, qy, qz, qw
    std::string truth_cuboid_file = base_folder+"/truth_objects.txt";// x, y, z, roll, pitch, yaw, longth, width, height
    std::string output_folder = base_folder+"/offline_cuboid/";
    std::string output_img_folder = base_folder+"/offline_img/";

    // Retrieve paths to images
    std::vector<int> vTimestamps;
    std::vector<std::string> vstrImageFilenames;
    std::vector<std::string> vstrDepthFilenames;
    std::vector<std::string> vstrBboxFilenames;
    dataset_tum data_loader;
    data_loader.LoadImages(rgb_file, vstrImageFilenames, vTimestamps);
    data_loader.LoadImages(depth_file, vstrDepthFilenames,vTimestamps);
    data_loader.LoadImages(bbox_2d_list_file, vstrBboxFilenames,vTimestamps);

    detect_cuboid_bbox object_detector;
    object_detector.whether_plot_detail_images = false;
    object_detector.whether_plot_ground_truth = false;
    object_detector.whether_plot_sample_images = false;
    object_detector.whether_plot_final_scores = true;
    object_detector.whether_sample_obj_dimension = false;
    object_detector.whether_sample_obj_yaw = true;
    object_detector.whether_add_plane_constraints = false;
    object_detector.whether_save_cam_obj_data = true;
    object_detector.whether_save_final_image = true;

    object_detector.Read_Kalib_TUM(calib_file);
    object_detector.Read_Camera_Pose_TUM(truth_camera_pose);
    object_detector.Read_Object_Info_TUM(truth_cuboid_file);

    int total_frame_number = vstrImageFilenames.size();
    // total_frame_number = 1;
    for (int frame_index = 0; frame_index < total_frame_number; frame_index++)
    {
        // frame_index=1487;
        std::cout << "-----------" << "frame index " << frame_index << " " << vstrImageFilenames[frame_index] << "-----------" << std::endl;

        //read image, calib and input
        ca::Profiler::tictoc("read image");
        std::string image_file = base_folder + "/" + vstrImageFilenames[frame_index];
        object_detector.Read_Image_TUM(image_file);
        std::string bbox_2d_file = base_folder + "/" + vstrBboxFilenames[frame_index];
        object_detector.Read_Bbox_2D_TUM(bbox_2d_file);
        ca::Profiler::tictoc("read image");

        // read depth image
        ca::Profiler::tictoc("plane estimation");
        PlaneDetection plane_detector;
        if(object_detector.whether_add_plane_constraints)
        {
          string depth_img_file = base_folder + "/" + vstrDepthFilenames[frame_index];
          plane_detector.setDepthValue(object_detector.mDepthMapFactor);
          plane_detector.setKalibValue(object_detector.Kalib);
          plane_detector.readDepthImage(depth_img_file);
          plane_detector.ConvertDepthToPointCloud();
          plane_detector.ComputePlanesFromOrganizedPointCloud();          
        }

        ca::Profiler::tictoc("plane estimation");

        std::vector<ObjectSet> frames_cuboids; // each 2d bbox generates an ObjectSet, which is vector of sorted proposals
        cv::Mat rgb_img = object_detector.rgb_img.clone();
        std::vector<cv::Mat> det_plane = plane_detector.mvPlaneCoefficients;
        Eigen::Matrix<double,4,4> transToWolrd;
		    Eigen::VectorXd cam_pose_Twc = object_detector.truth_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
        // std::cout << "cam_pose_Twc: \n" << cam_pose_Twc.transpose() << std::endl;
        transToWolrd.setIdentity();
        transToWolrd.block(0,0,3,3) = Eigen::Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
        transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
        std::cout << "transToWolrd: \n" << transToWolrd << std::endl;

        // create save file every frame

        string output_cuboid_file = output_folder + "/" + std::to_string(frame_index) + "_3d_cuboids.txt";
        string output_cuboid_img = output_img_folder + "/" + std::to_string(frame_index) + "_3d_img.png";
        object_detector.detect_cuboid_every_frame(rgb_img, det_plane, transToWolrd, frames_cuboids, output_cuboid_file, output_cuboid_img);
        /*
		int det_obj_num = object_detector.detected_obj_name.size();
		std::vector<cuboid *> single_object_candidate;
		for (size_t object_id = 0; object_id < det_obj_num; object_id++)
		{
			double det_obj_yaw; // do not need
			std::string det_obj_name;
			Eigen::Vector4d det_obj_2d_bbox;
			Eigen::Vector3d det_obj_dim_ave;
			bool check_object_name = object_detector.Get_Object_Input_SUNRGBD(int(object_id), det_obj_name, det_obj_2d_bbox, det_obj_dim_ave, det_obj_yaw);
			if (!check_object_name) // exclude object that we do not care
				continue;
			else
			{
				ca::Profiler::tictoc("object detection");
				cv::Mat rgb_img = object_detector.rgb_img.clone();
				std::vector<cv::Mat> det_plane = plane_detector.mvPlaneCoefficients;
				object_detector.detect_cuboid_with_bbox_constraints(rgb_img, det_obj_2d_bbox, det_obj_yaw, det_obj_dim_ave, det_plane, single_object_candidate);
				ca::Profiler::tictoc("object detection");
			} // loop object name check

			// // print select info
			// for (size_t i = 0; i < single_object_candidate.size(); i++)
			// {
			// 	cuboid* detected_cuboid = single_object_candidate[i];
			// 	Eigen::Vector3d obj_dim_saved = detected_cuboid->obj_dim_camera;
			// 	Eigen::Matrix4d obj_mat_final_camera = obj_mat_multi_final_camera[i];
			// 	Eigen::Matrix4d obj_mat_final_world = transToWolrd * obj_mat_final_camera;
			// 	Eigen::Vector3d obj_loc_saved = obj_mat_final_world.col(3).head(3);
			// 	Eigen::Matrix3d mat_temp = obj_mat_final_world.block(0, 0, 3, 3);
			// 	Eigen::Vector3d obj_rpy_saved;
			// 	quat_to_euler_zyx(Quaterniond(mat_temp), obj_rpy_saved(0), obj_rpy_saved(1), obj_rpy_saved(2));
			// 	std::cout << "frame: " << frame_index << " object: " << truth_class[i]
			// 			  << "\nobj_loc_saved: " << obj_loc_saved.transpose()
			// 			  << "\nobj_rpy_saved: " << obj_rpy_saved.transpose()
			// 			  << "\nobj_dim_saved: " << obj_dim_saved.transpose()
			// 			  << std::endl;
			// }

			if (object_detector.whether_plot_final_scores || object_detector.whether_save_final_image)
			{
				cv::Mat plot_img = rgb_img.clone();
				for (size_t i = 0; i < single_object_candidate.size(); i++)
				{
					cuboid* detected_cuboid = single_object_candidate[i];
					Eigen::Vector3d obj_dim_plot = detected_cuboid->obj_dim_camera;
					Eigen::Vector3d obj_loc_plot = detected_cuboid->obj_loc_camera;
					Eigen::Matrix3d obj_rot_plot = detected_cuboid->obj_rot_camera;
					Eigen::Matrix3d Kalib = object_detector.Kalib;
					plot_3d_box_with_loc_dim_camera(plot_img, Kalib, obj_loc_plot, obj_dim_plot, obj_rot_plot);
				}
				if (object_detector.whether_plot_final_scores)
				{
					cv::imshow("final selection image", plot_img);
					cv::waitKey(0);
				}
				if (object_detector.whether_save_final_image)
				{
					string save_img_file = base_folder + "/offline_img/" + vstrImageFilenames[frame_index] + "_3d_img.png";
					cv::imwrite(save_img_file, plot_img);
				}
			}
			if (object_detector.whether_save_cam_obj_data)
			{
				for (size_t i = 0; i < single_object_candidate.size(); i++)
				{
					cuboid* detected_cuboid = single_object_candidate[i];
					Eigen::Vector4d raw_2d_objs = detected_cuboid->bbox_2d;
					// Eigen::Vector3d obj_dim_cam = obj_dim_multi_final_camera[i];
					// Eigen::Matrix4d obj_mat_final_camera = obj_mat_multi_final_camera[i];
					// Eigen::Vector3d obj_loc_cam = obj_mat_final_camera.col(3).head(3);
					// Eigen::Matrix3d obj_ori_cam = obj_mat_final_camera.block(0, 0, 3, 3);
					Eigen::Vector3d obj_dim_plot = detected_cuboid->obj_dim_camera;
					Eigen::Vector3d obj_loc_plot = detected_cuboid->obj_loc_camera;
					Eigen::Matrix3d obj_rot_plot = detected_cuboid->obj_rot_camera;
					// Eigen::Vector3d obj_rpy_cam;
					// quat_to_euler_zyx(Quaterniond(obj_ori_cam), obj_rpy_cam(0), obj_rpy_cam(1), obj_rpy_cam(2));
					// std::cout << "obj_mat_final_camera: " << obj_mat_final_camera << std::endl;
					// transfer to global coordinates
					Eigen::Vector3d obj_dim_world = detected_cuboid->obj_dim_camera;
					Eigen::Matrix4d obj_mat_final_camera;
					obj_mat_final_camera.setIndentity();
					obj_mat_final_camera.block(0, 0, 3, 3) = detected_cuboid->obj_rot_camera;
					obj_mat_final_camera.col(3).head(3) = detected_cuboid->obj_loc_camera;
					Eigen::Matrix4d obj_mat_final_world = transToWolrd * obj_mat_final_camera;
					Eigen::Vector3d obj_loc_world = obj_mat_final_world.col(3).head(3);
					Eigen::Matrix3d mat_temp = obj_mat_final_world.block(0, 0, 3, 3);
					Eigen::Vector3d obj_rpy_world;
					quat_to_euler_zyx(Quaterniond(mat_temp), obj_rpy_world(0), obj_rpy_world(1), obj_rpy_world(2));
					std::cout << "esti global_location: " << obj_loc_world.transpose() << std::endl;
					std::cout << "esti global_orient: " << obj_rpy_world.transpose() << std::endl;
					// pay attention to the format
					online_stream_cube_multi << obj_final_name[i] << " " << raw_2d_objs(0)
												<< " " << raw_2d_objs(1) << " " << raw_2d_objs(2) << " " << raw_2d_objs(3)
												<< " " << obj_loc_world(0) << " " << obj_loc_world(1) << " " << obj_loc_world(2)
												<< " " << obj_rpy_world(0) << " " << obj_rpy_world(1) << " " << obj_rpy_world(2)
												<< " " << obj_dim_world(1) << " " << obj_dim_world(0) << " " << obj_dim_world(2)
												<< " " << "\n";

						// // save local value
						// online_stream_cube_multi << bbox_class[i] << " " << obj_loc_cam(0)
						//     << " " << obj_loc_cam(1)  << " " << obj_loc_cam(2)
						//     << " " << obj_rpy_cam(0) << " " << obj_rpy_cam(1) << " " << obj_rpy_cam(2)
						//     << " " << obj_dim_cam(0) << " " << obj_dim_cam(1) << " " << obj_dim_cam(2)
						//     << " " << raw_2d_objs(i,0) << " " << raw_2d_objs(i,1) << " " << raw_2d_objs(i,2) << " " << raw_2d_objs(i,3)
						//     << " " << obj_final_score[i] << " " << "\n";

						// // save ground truth
						// online_stream_cube_multi << bbox_class[i] << " " << truth_cuboid_frame(i,0)
						//     << " " << truth_cuboid_frame(i,1)  << " " << truth_cuboid_frame(i,2)
						//     << " " << truth_cuboid_frame(i,3) << " " << truth_cuboid_frame(i,4) << " " << truth_cuboid_frame(i,5)
						//     << " " << truth_cuboid_frame(i,6) << " " << truth_cuboid_frame(i,7) << " " << truth_cuboid_frame(i,8)
						//     << " " << "\n";

					} // if has final bbox
				}	  // loop for object_id
				online_stream_cube_multi.close();
			}

		} // loop object_id
*/

        /*
      if(!object_detector.truth_cuboid_list.isZero()) // otherwise, no detection
      {
        if(whether_plot_ground_truth)
        {
          cv::Mat plot_2d_img = rgb_img.clone();
          cv::Mat plot_3d_img = rgb_img.clone();
          for (size_t obj_id = 0; obj_id < truth_cuboid_list.rows(); obj_id++)
          {
              Eigen::Vector4d raw_2d_objs = truth_cuboid_list.row(obj_id).head(4);
              Eigen::Vector3d raw_centroid = truth_cuboid_list.row(obj_id).segment(4,3);
              Eigen::Vector3d raw_dimension = truth_cuboid_list.row(obj_id).segment(7,3);
              Eigen::Vector2d heading = truth_cuboid_list.row(obj_id).tail(2);
              double yaw = 1 * atan2(heading(1), heading(0));
              // std::cout << "raw_2d_objs: " << raw_2d_objs.transpose() << std::endl;
              // std::cout << "raw_centroid: " << raw_centroid.transpose() << std::endl;
              // std::cout << "raw_dimension: " << raw_dimension.transpose() << std::endl;
              // std::cout << "yaw: " << yaw << " heading: " << heading.transpose() << std::endl;
                
              Eigen::MatrixXd plot_bbox(1,4);
              plot_bbox << raw_2d_objs;
              plot_2d_bbox_with_xywh(plot_2d_img,plot_bbox);

              // double plot_yaw = yaw;
              // Eigen::Vector3d plot_loc_world = raw_centroid;
              // Eigen::Vector3d plot_dim_world;
              // plot_dim_world << raw_dimension(1), raw_dimension(0), raw_dimension(2);
              // Eigen::MatrixXd plot_proj =  proj_matrix; 
              // plot_3d_box_with_loc_dim_world(plot_3d_img, plot_proj, plot_loc_world, plot_dim_world, plot_yaw);

              Eigen::Matrix4d object_matrix;
              object_matrix.setIdentity();
              object_matrix.block(0,0,3,3) = euler_zyx_to_rot<double>(0, 0, yaw);
              object_matrix.col(3).head(3) = raw_centroid;
              Eigen::Matrix4d object_matrix_camera = transToWolrd.inverse()*object_matrix;
              Eigen::Matrix3d obj_rot_camera = object_matrix_camera.block(0,0,3,3);
              Eigen::Vector3d obj_loc_camera = object_matrix_camera.col(3).head(3);
              Eigen::Vector3d obj_dim_camera = Eigen::Vector3d(raw_dimension(0), raw_dimension(1), raw_dimension(2));
              std::cout << "object_matrix_camera: \n" << object_matrix_camera << std::endl;
              // plot_2d_bbox_with_xywh(plot_3d_img,raw_2d_objs);
              plot_3d_box_with_loc_dim_camera(plot_3d_img, Kalib, obj_loc_camera, obj_dim_camera, obj_rot_camera);

              
          }
          cv::imshow("2d bounding box", plot_2d_img);
          cv::imshow("3d cuboid ground truth", plot_3d_img);
          cv::waitKey(0);
        }

        std::vector<Eigen::Matrix4d> obj_mat_multi_final_camera;
        std::vector<Eigen::Vector3d> obj_dim_multi_final_camera;
        std::vector<Eigen::Vector4d> obj_bbox_multi_final_camera;
        std::vector<string> obj_final_name;
        std::vector<double> obj_final_score;
        for (size_t object_id = 0; object_id < truth_cuboid_list.rows(); object_id++)
        {
          // prepare ground truth, we only need dimension
          std::string obj_name_tmp = truth_class[object_id];
          Eigen::Vector4d raw_2d_objs = truth_cuboid_list.row(object_id).head(4);
          Eigen::Vector3d object_loc_global = truth_cuboid_list.row(object_id).segment(4,3);
          Eigen::Vector3d object_dim_truth = truth_cuboid_list.row(object_id).segment(7,3);
          Eigen::Vector3d object_dim_global;
          object_dim_global << object_dim_truth(1), object_dim_truth(0), object_dim_truth(2);
          Eigen::Vector2d heading = truth_cuboid_list.row(object_id).tail(2);
          double object_yaw = 1 * atan2(heading(1), heading(0));

          if(raw_2d_objs(0)>rgb_img.cols || raw_2d_objs(0)+raw_2d_objs(2)>rgb_img.cols 
              || raw_2d_objs(1)>rgb_img.rows ||raw_2d_objs(1)+raw_2d_objs(3)>rgb_img.rows)
            continue;

          // read dimension from file
          auto it = std::find(obj_name_list.begin(), obj_name_list.end(), truth_class[object_id]); // find the name, if not, ignore
          if (it == obj_name_list.end()) //other object, ignore
          {
            std::cout << "----- class name ignore -----" << std::endl;
            continue;
          }
          else
          {
            int obj_name_id = std::distance(obj_name_list.begin(), it);
            object_dim_global << obj_dim_list(obj_name_id,2), obj_dim_list(obj_name_id,1), obj_dim_list(obj_name_id,3);
          }
          

          // when there are more than one object, it would be good to sample object yaw
          std::vector<double> obj_length_samples;
          std::vector<double> obj_width_samples;
          std::vector<double> obj_height_samples;
          std::vector<double> obj_yaw_samples;
          double s_range = 0.2;
          double s_step = 0.2;
          if(whether_sample_obj_dimension)
          {
            linespace<double>(object_dim_global(0)*(1-s_range), object_dim_global(0)*(1+s_range), object_dim_global(0)*s_step, obj_length_samples);
            linespace<double>(object_dim_global(1)*(1-s_range), object_dim_global(1)*(1+s_range), object_dim_global(1)*s_step, obj_width_samples);
            linespace<double>(object_dim_global(2)*(1-s_range), object_dim_global(2)*(1+s_range), object_dim_global(2)*s_step, obj_height_samples);
          }
          else
          {
            obj_length_samples.push_back(object_dim_global(0));
            obj_width_samples.push_back(object_dim_global(1));
            obj_height_samples.push_back(object_dim_global(2));
          }

          // sample object yaw, note the yaw is in world coordinates, could we sample local yaw?
          if(whether_sample_obj_yaw)
          {
            double yaw_init = camera_rpy(2) - 90.0 / 180.0 * M_PI; // yaw init is directly facing the camera, align with camera optical axis
            linespace<double>(yaw_init - 90.0 / 180.0 * M_PI, yaw_init + 90.0 / 180.0 * M_PI, 5.0 / 180.0 * M_PI, obj_yaw_samples);
            // linespace<double>(yaw_init - 180.0 / 180.0 * M_PI, yaw_init + 180.0 / 180.0 * M_PI, 5.0 / 180.0 * M_PI, obj_yaw_samples);
          }
          else
            obj_yaw_samples.push_back(object_yaw);

          // // prepare for score, do not need loop every proposal, outside
          // // prepare 1, compute canny and distance map for distance error
          //   // TODO could canny or distance map outside sampling height to speed up!!!!   Then only need to compute canny onces.
          //   // detect canny edges and compute distance transform  NOTE opencv canny maybe different from matlab. but roughly same
          double bbox_thres = 30.0;
          Eigen::Vector4d bbox_canny;
          bbox_canny(0) = std::max(0.0, raw_2d_objs(0)-bbox_thres);
          bbox_canny(1) = std::max(0.0, raw_2d_objs(1)-bbox_thres);
          bbox_canny(2) = std::min(double(rgb_img.cols), raw_2d_objs(2)+raw_2d_objs(0)+bbox_thres);
          bbox_canny(3) = std::min(double(rgb_img.rows), raw_2d_objs(3)+raw_2d_objs(1)+bbox_thres);
          // std::cout << "bbox_canny: " << bbox_canny.transpose() << std::endl;
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
          double pre_merge_dist_thre = 10;
          double pre_merge_angle_thre = 5;
          double edge_length_threshold = 40;
          MatrixXd all_lines_merge_inobj;
          merge_break_lines(all_lines_inside_object.topRows(inside_obj_edge_num), all_lines_merge_inobj, pre_merge_dist_thre,
                    pre_merge_angle_thre, edge_length_threshold);
          // std::cout << "all_lines_merge_inobj: " << all_lines_merge_inobj << std::endl;
          filter_line_by_coincide(all_lines_merge_inobj, pre_merge_angle_thre, pre_merge_dist_thre);
          // std::cout << "all_lines_merge_inobj: " << all_lines_merge_inobj << std::endl;
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
          Eigen::Matrix4d obj_mat_final_camera;
          obj_mat_final_camera.setZero();
          Eigen::Vector3d obj_dim_final_camera;
          Eigen::Vector4d obj_bbox_final_camera;

              // cv::Mat plot_img = rgb_img.clone();

          // if there are more than one object, it would be better to sample object yaw
          for (int obj_len_id = 0; obj_len_id < obj_length_samples.size(); obj_len_id++)
            for (int obj_wid_id = 0; obj_wid_id < obj_width_samples.size(); obj_wid_id++)
              for (int obj_hei_id = 0; obj_hei_id < obj_height_samples.size(); obj_hei_id++)
                for (int obj_yaw_id = 0; obj_yaw_id < obj_yaw_samples.size(); obj_yaw_id++)
          {
            std::cout <<"yaw_id: " << obj_yaw_id << "-----------" << std::endl;

            // step 1: get object dimension and orintation in camera coordinate
            Eigen::Vector3d object_dim_cam; // should change coordinates?
            object_dim_cam(0) = obj_length_samples[obj_len_id];
            object_dim_cam(1) = obj_width_samples[obj_wid_id];
            object_dim_cam(2) = obj_height_samples[obj_hei_id];
            std::cout << "object_dim_cam: " << object_dim_cam.transpose() << std::endl;

            Eigen::Matrix3d Rot_Mat_new = euler_zyx_to_rot(camera_rpy(0), camera_rpy(1), camera_rpy(2));
            transToWolrd.block(0,0,3,3) = Rot_Mat_new;
            Eigen::Matrix3d obj_local_rot;
            double yaw_sample = obj_yaw_samples[obj_yaw_id];
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

            Eigen::Vector3d esti_location;
            esti_location.setZero();
            calculate_location_new(object_dim_cam, cam_to_img, box_2d, obj_local_rot, transToWolrd, theta_ray, esti_location);
            std::cout << "esti location: " << esti_location.transpose() << std::endl;
            
            if(esti_location.isZero())
            {
              std::cout << " bbox estimation failed " << std::endl;
              continue;
            }
  
            // step 3: compute visible corners and edge, prepare for score
            // prepare for 2d corner in image
            MatrixXd corner_3d = compute3D_BoxCorner_in_camera(object_dim_cam, esti_location, obj_local_rot);
            MatrixXd box_corners_2d_float = project_camera_points_to_2d(corner_3d, Kalib);
            Eigen::Vector4d bbox_new;
            bbox_new(0) = std::max(0.0, box_corners_2d_float.row(0).minCoeff());
            bbox_new(1) = std::max(0.0, box_corners_2d_float.row(1).minCoeff());
            bbox_new(2) = std::min(double(rgb_img.cols), box_corners_2d_float.row(0).maxCoeff());
            bbox_new(3) = std::min(double(rgb_img.rows), box_corners_2d_float.row(1).maxCoeff());
            // bbox_new << box_corners_2d_float.row(0).minCoeff(), box_corners_2d_float.row(1).minCoeff(),
            //             box_corners_2d_float.row(0).maxCoeff(), box_corners_2d_float.row(1).maxCoeff();
            std::cout << "bbox_new: " << bbox_new.transpose() << std::endl;          

            // prepare for visible 2d corner in image
            // based on trial, may be different from different dataset ...
            Eigen::MatrixXi visible_edge_pt_ids;

            if (object_rpy(2) >= -180.0/180.0*M_PI && object_rpy(2) < -170.0/180.0*M_PI) 
            {
              visible_edge_pt_ids.resize(7, 2);
              visible_edge_pt_ids << 0,1, 1,2, 2,3, 3,0, 0,4, 3,7, 6,7; // 0123 on th top are shown all the time
            }
            else if (object_rpy(2) >= -170.0/180.0*M_PI && object_rpy(2) < -90.0/180.0*M_PI)
            {
              visible_edge_pt_ids.resize(9, 2);
              visible_edge_pt_ids << 0,1, 1,2, 2,3, 3,0, 0,4, 1,5, 3,7, 4,5, 4,7; // 0123 on th top are shown all the time
            }
            else if (object_rpy(2) >= -90.0/180.0*M_PI && object_rpy(2) < -80.0/180.0*M_PI) 
            {
              visible_edge_pt_ids.resize(7, 2);
              visible_edge_pt_ids << 0,1, 1,2, 2,3, 3,0, 0,4, 3,7, 4,7; // 0123 on th top are shown all the time
            }
            else if (object_rpy(2) >= -80.0/180.0*M_PI && object_rpy(2) < 0.0/180.0*M_PI) 
            {
              visible_edge_pt_ids.resize(9, 2);
              visible_edge_pt_ids << 0,1, 1,2, 2,3, 3,0, 0,4, 2,6, 3,7, 5,7, 6,7; // 0123 on th top are shown all the time
            }
            else if (object_rpy(2) >= 0.0/180.0*M_PI && object_rpy(2) < 10.0/180.0*M_PI)
            {
              visible_edge_pt_ids.resize(7, 2);
              visible_edge_pt_ids << 0,1, 1,2, 2,3, 3,0, 2,6, 3,7, 6,7; // 0123 on th top are shown all the time
            }
            else if (object_rpy(2) >= 10.0/180.0*M_PI && object_rpy(2) < 90.0/180.0*M_PI) 
            {
              visible_edge_pt_ids.resize(9, 2);
              visible_edge_pt_ids << 0,1, 1,2, 2,3, 3,0, 1,5, 2,6, 3,7, 5,6, 6,7; // 0123 on th top are shown all the time
            }
            else if (object_rpy(2) >= 90.0/180.0*M_PI && object_rpy(2) < 100.0/180.0*M_PI) // -3.14=+3.14
            {
              visible_edge_pt_ids.resize(7, 2);
              visible_edge_pt_ids << 0,1, 1,2, 2,3, 3,0, 2,6, 3,7, 6,7; // 0123 on th top are shown all the time
            }
            else if (object_rpy(2) >= 100.0/180.0*M_PI && object_rpy(2) < 180.0/180.0*M_PI) 
            {
              visible_edge_pt_ids.resize(9, 2);
              visible_edge_pt_ids << 0,1, 1,2, 2,3, 3,0, 0,4, 2,6, 3,7, 4,7, 6,7; // 0123 on th top are shown all the time
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

            // step 4: add score function : distance error and angle error
            // make sure new bbox are in the canny image, else, distance error is not accurate
            if( bbox_new(0) >= bbox_canny(0) && bbox_new(1) >= bbox_canny(1) &&   // xmin, ymin
                bbox_new(2) <= bbox_canny(2) && bbox_new(3) <= bbox_canny(3) ) // xmax, ymax
            {
            // compute distance error
            bool reweight_edge_distance = false; // if want to compare with all configurations. we need to reweight
            double sum_dist=0.0;
            MatrixXd box_corners_2d_float_shift(2, 8); // shift from whole image to canny image
            box_corners_2d_float_shift.row(0) = box_corners_2d_float.row(0).array() - bbox_canny(0);
            box_corners_2d_float_shift.row(1) = box_corners_2d_float.row(1).array() - bbox_canny(1);
            sum_dist = box_edge_sum_dists(dist_map, box_corners_2d_float_shift, visible_edge_pt_ids, reweight_edge_distance);
            double obj_diag_length = sqrt(raw_2d_objs(2) * raw_2d_objs(2) + raw_2d_objs(3) * raw_2d_objs(3)); // sqrt(width^2+height^2)
            double distance_error = std::min(fabs(sum_dist/obj_diag_length), 1e9);
            // std::cout <<"sum_dist: " << sum_dist << " distance_error: " << distance_error << std::endl;

            // compute angle error
            double angle_error = box_edge_angle_error(all_lines_merge_inobj, box_edges_visible);
            // double angle_error = box_edge_angle_error_new(all_lines_merge_inobj, box_edges_visible);
            // std::cout << "angle_error: " << angle_error << std::endl;

            // compute feature error = (dis_err + k*angle_err)/(1+k)
            double weight_angle = 0.5; 
            double feature_error = (1-weight_angle)*(distance_error/10.0) + weight_angle * (fmod(angle_error,M_PI)/M_PI);

            // compute obj-plane error
            double obj_plane_error = compute_obj_plane_error(esti_location, object_dim_cam, obj_local_rot, mvPlaneCoefficients);

            double weight_3d = 0.8; 
            if(!whether_add_plane_constraints)
              weight_3d = 0.0;
            combined_scores = (1-weight_3d)* feature_error+ weight_3d * obj_plane_error;

            std::cout << "feature_error: " << feature_error << " obj_plane_error: " << obj_plane_error << std::endl;
            std::cout << "combined_scores: " << combined_scores << std::endl;
            
            }// loop if(in 2d box)
            else
            {
              std::cout << "combined_scores: 10000, outside bbox"  << std::endl;
            } // loop if(in 2d box)

            // step 5: update selected cuboid with the min_error (distance error + angle error)
            if (combined_scores < min_com_error)
            {
              std::cout << "yaw update, combined_scores: " << combined_scores << " min_com_error: "<< min_com_error << std::endl;  
              min_com_error = combined_scores;
              obj_mat_final_camera.setIdentity();
              obj_mat_final_camera.block(0,0,3,3) = obj_local_rot;
              obj_mat_final_camera.col(3).head(3) = esti_location;
              obj_dim_final_camera = object_dim_cam;
              obj_bbox_final_camera = Eigen::Vector4d(bbox_new(0), bbox_new(1), bbox_new(2)-bbox_new(0), bbox_new(3)-bbox_new(1));
            }
            // plot in camera coordinate
            if(whether_plot_sample_images)
            {
              cv::Mat plot_img = rgb_img.clone();
              plot_3d_box_with_loc_dim_camera(plot_img, Kalib, esti_location, object_dim_cam, obj_local_rot);
              cv::imshow("proposal image", plot_img);
              cv::waitKey(0);
            }
          } // loop yaw_id

          obj_mat_multi_final_camera.push_back(obj_mat_final_camera);
          obj_dim_multi_final_camera.push_back(obj_dim_final_camera);
          obj_bbox_multi_final_camera.push_back(raw_2d_objs);
          obj_final_name.push_back(obj_name_tmp);
          obj_final_score.push_back(min_com_error);

		      ca::Profiler::tictoc("object detection");
          if(whether_plot_final_scores)
          {
            Eigen::Matrix3d plot_obj_rot = obj_mat_final_camera.block(0,0,3,3);
            Eigen::Vector3d plot_obj_loc = obj_mat_final_camera.col(3).head(3);
            Eigen::Vector3d plot_obj_dim = obj_dim_final_camera;
            std::cout << "!!!!!!!final_location_camera:"  << plot_obj_loc.transpose()  << std::endl;
            cv::Mat plot_img = rgb_img.clone();
            plot_3d_box_with_loc_dim_camera(plot_img, Kalib, plot_obj_loc, plot_obj_dim, plot_obj_rot);
            cv::imshow("selection image", plot_img);
            cv::waitKey(0);              
          }


        } // loop object id

        // print select info
        for (size_t i = 0; i < obj_mat_multi_final_camera.size(); i++)
        {
          Eigen::Vector3d obj_dim_saved = obj_dim_multi_final_camera[i];
          Eigen::Matrix4d obj_mat_final_camera = obj_mat_multi_final_camera[i];
          Eigen::Matrix4d obj_mat_final_world = transToWolrd * obj_mat_final_camera;
          Eigen::Vector3d obj_loc_saved = obj_mat_final_world.col(3).head(3);
          Eigen::Matrix3d mat_temp = obj_mat_final_world.block(0,0,3,3);
          Eigen::Vector3d obj_rpy_saved;
          quat_to_euler_zyx(Quaterniond(mat_temp), obj_rpy_saved(0), obj_rpy_saved(1), obj_rpy_saved(2));
          std::cout << "frame: " << frame_index << " object: " << truth_class[i]
                    << "\nobj_loc_saved: " << obj_loc_saved.transpose()
                    << "\nobj_rpy_saved: " << obj_rpy_saved.transpose()
                    << "\nobj_dim_saved: " << obj_dim_saved.transpose()
                    << std::endl;
        }
        if(whether_plot_final_scores||whether_save_final_image)
        {
          cv::Mat plot_img = rgb_img.clone();
          for (size_t i = 0; i < obj_mat_multi_final_camera.size(); i++)
          {
            Eigen::Vector3d obj_dim_plot = obj_dim_multi_final_camera[i];
            Eigen::Matrix4d obj_mat_plot = obj_mat_multi_final_camera[i];
            Eigen::Vector3d obj_loc_plot = obj_mat_plot.col(3).head(3);
            Eigen::Matrix3d obj_rot_plot = obj_mat_plot.block(0,0,3,3);
            plot_3d_box_with_loc_dim_camera(plot_img, Kalib, obj_loc_plot, obj_dim_plot, obj_rot_plot);
          }
          if(whether_plot_final_scores)
          {
            cv::imshow("final selection image", plot_img);
            cv::waitKey(0);            
          }
          if(whether_save_final_image)
          {
            string save_img_file = base_folder + "/offline_img/"+vstrImageFilenames[frame_index]+"_3d_img.png";
            cv::imwrite(save_img_file, plot_img);
          }

        }
        if(whether_save_cam_obj_data)
        {
          for (size_t i = 0; i < obj_mat_multi_final_camera.size(); i++)
          {
            if(obj_mat_multi_final_camera[i](3,3)!=0.0)
            {
              // Eigen::Vector4d raw_2d_objs = obj_bbox_multi_final_camera[i];
              Eigen::Vector4d raw_2d_objs = obj_bbox_multi_final_camera[i];
              Eigen::Vector3d obj_dim_cam = obj_dim_multi_final_camera[i];
              Eigen::Matrix4d obj_mat_final_camera = obj_mat_multi_final_camera[i];
              Eigen::Vector3d obj_loc_cam = obj_mat_final_camera.col(3).head(3);
              Eigen::Matrix3d obj_ori_cam = obj_mat_final_camera.block(0,0,3,3);
              Eigen::Vector3d obj_rpy_cam;
              quat_to_euler_zyx(Quaterniond(obj_ori_cam), obj_rpy_cam(0), obj_rpy_cam(1), obj_rpy_cam(2));
              std::cout << "obj_mat_final_camera: " << obj_mat_final_camera << std::endl;
              // transfer to global coordinates 
              Eigen::Vector3d obj_dim_world = obj_dim_multi_final_camera[i];
              Eigen::Matrix4d obj_mat_final_world = transToWolrd * obj_mat_final_camera;
              Eigen::Vector3d obj_loc_world = obj_mat_final_world.col(3).head(3);
              Eigen::Matrix3d mat_temp = obj_mat_final_world.block(0,0,3,3);
              Eigen::Vector3d obj_rpy_world;
              quat_to_euler_zyx(Quaterniond(mat_temp), obj_rpy_world(0), obj_rpy_world(1), obj_rpy_world(2));
              std::cout << "esti global_location: " << obj_loc_world.transpose() << std::endl;
              std::cout << "esti global_orient: " << obj_rpy_world.transpose() << std::endl;
              // pay attention to the format
              online_stream_cube_multi << obj_final_name[i] << " " << raw_2d_objs(0)
                  << " " << raw_2d_objs(1) << " " << raw_2d_objs(2) << " " << raw_2d_objs(3) 
                  << " " << obj_loc_world(0) << " " << obj_loc_world(1) << " " << obj_loc_world(2) 
                  << " " << obj_rpy_world(0) << " " << obj_rpy_world(1) << " " << obj_rpy_world(2)
                  << " " << obj_dim_world(1) << " " << obj_dim_world(0) << " " << obj_dim_world(2)
                  << " " << "\n";

              // // save local value
              // online_stream_cube_multi << bbox_class[i] << " " << obj_loc_cam(0)
              //     << " " << obj_loc_cam(1)  << " " << obj_loc_cam(2) 
              //     << " " << obj_rpy_cam(0) << " " << obj_rpy_cam(1) << " " << obj_rpy_cam(2)
              //     << " " << obj_dim_cam(0) << " " << obj_dim_cam(1) << " " << obj_dim_cam(2)
              //     << " " << raw_2d_objs(i,0) << " " << raw_2d_objs(i,1) << " " << raw_2d_objs(i,2) << " " << raw_2d_objs(i,3)
              //     << " " << obj_final_score[i] << " " << "\n";

              // // save ground truth
              // online_stream_cube_multi << bbox_class[i] << " " << truth_cuboid_frame(i,0)
              //     << " " << truth_cuboid_frame(i,1)  << " " << truth_cuboid_frame(i,2) 
              //     << " " << truth_cuboid_frame(i,3) << " " << truth_cuboid_frame(i,4) << " " << truth_cuboid_frame(i,5)
              //     << " " << truth_cuboid_frame(i,6) << " " << truth_cuboid_frame(i,7) << " " << truth_cuboid_frame(i,8)
              //     << " " << "\n";

            } // if has final bbox
          } // loop for object_id
          online_stream_cube_multi.close();
        }


      } // loop if(2d_bbox)

      else // else no bbox
      {
        std::cout << "++++++++++++ NO BBOX ++++++++"  << std::endl;
      }
      */
    } // loop frame_id

    ca::Profiler::print_aggregated(std::cout);

    return 0;
}