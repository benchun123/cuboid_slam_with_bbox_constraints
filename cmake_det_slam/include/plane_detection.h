#ifndef PLANE_DETECTION_H
#define PLANE_DETECTION_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <Eigen/Eigen>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>


#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl/features/normal_3d.h>  // include pcl::NormalEstimation
#include <pcl/segmentation/organized_multi_plane_segmentation.h> // include pcl::OrganizedMultiPlaneSegmentation
#include <pcl/features/integral_image_normal.h>  // include pcl::IntegralImageNormalEstimation
#include <pcl/segmentation/region_growing.h>// include pcl::RegionGrowing
// #include <pcl/visualization/cloud_viewer.h>//include pcl::visualization::CloudViewer
#include <boost/make_shared.hpp>

using namespace std;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud <PointT> PointCloud;

class PlaneDetection
{
public:
	// ImagePointCloud cloud;
  	pcl::PointCloud<pcl::PointXYZRGB> rgbd_cloud;
	cv::Mat seg_img_; // segmentation image
	cv::Mat imDepth; // input depth image
	cv::Mat color_img_; // input color image
	int plane_num_;
	int imgWidth, imgHeight;	// image size
	double fx, fy, cx, cy;  // Camera intrinsic parameters.
	double mDepthMapFactor; // scale coordinate unit in mm
	std::vector<cv::Mat> mvPlaneCoefficients; // plane normal 
	std::vector<pcl::PointCloud <PointT> > mvPlanePoints; // plane points
    std::vector<pcl::PointCloud <pcl::PointXYZRGB> > mvBoundaryPoints;

	int MultiPlane_SizeMin;
	float MultiPlane_AngleThre;
	float MultiPlane_DistThre;


public:
	PlaneDetection();

	~PlaneDetection();

	bool readIntrinsicParameterFile(string& filename);
	bool setDepthValue(double value);
	bool setKalibValue(Eigen::Matrix3d& kalib);
	bool readColorImage(string& filename);
	bool readDepthImage(string& filename);
	bool ConvertDepthToPointCloud();

	void ComputePlanesFromOrganizedPointCloud();
};

#endif
