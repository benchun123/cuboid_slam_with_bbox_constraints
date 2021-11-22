#include "plane_detection.h"

PlaneDetection::PlaneDetection()
{
}

PlaneDetection::~PlaneDetection()
{
	color_img_.release();
	seg_img_.release();
	imDepth.release();
}

bool PlaneDetection::readIntrinsicParameterFile(string &filename)
{
	std::cout << "start reading calib " << std::endl;
	cv::FileStorage fSettings(filename, cv::FileStorage::READ);

	imgWidth = fSettings["Camera.width"];
	imgHeight = fSettings["Camera.height"];
	fx = fSettings["Camera.fx"];
	fy = fSettings["Camera.fy"];
	cx = fSettings["Camera.cx"];
	cy = fSettings["Camera.cy"];
	mDepthMapFactor = fSettings["DepthMapFactor"];

	if (fabs(mDepthMapFactor) < 1e-5)
		mDepthMapFactor = 1;
	else
		mDepthMapFactor = 1.0f / mDepthMapFactor;
	cout << "- fx: " << fx << endl;
	cout << "- fy: " << fy << endl;
	cout << "- cx: " << cx << endl;
	cout << "- cy: " << cy << endl;
	cout << "- mDepthMapFactor: " << mDepthMapFactor << endl;

	// MultiPlane_SizeMin = fSettings["MultiPlane.SizeMin"];
	// MultiPlane_AngleThre = fSettings["MultiPlane.AngleThre"];
	// MultiPlane_DistThre = fSettings["MultiPlane.DistThre"];

	return true;
}

bool PlaneDetection::setKalibValue(Eigen::Matrix3d &kalib)
{
	fx = kalib(0, 0);
	fy = kalib(1, 1);
	cx = kalib(0, 2);
	cy = kalib(1, 2);
	return true;
}

bool PlaneDetection::setDepthValue(double value)
{
	mDepthMapFactor = value;
	cout << "- mDepthMapFactor: " << mDepthMapFactor << endl;
	if (fabs(mDepthMapFactor) < 1e-5)
		mDepthMapFactor = 1;
	else
		mDepthMapFactor = 1.0f / mDepthMapFactor;
	return true;
}

bool PlaneDetection::readColorImage(string &filename)
{
	std::cout << "start reading rgb image " << std::endl;
	color_img_ = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	if (color_img_.empty() || color_img_.depth() != CV_8U)
	{
		cout << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
		return false;
	}
	return true;
}

bool PlaneDetection::readDepthImage(string &filename)
{
	std::cout << "start reading depth image " << std::endl;
	// cv::Mat imDepth = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH);
	imDepth = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
	if (imDepth.empty() || imDepth.depth() != CV_16U)
	{
		cout << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
		return false;
	}
	// // change depth value, do not know why, but do it
	if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
		imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);
}

bool PlaneDetection::ConvertDepthToPointCloud()
{
	// // translate to point cloud
	// // PointCloud::Ptr inputCloud( new PointCloud() );
	int cloudDis = 1;
	for (int m = 0; m < imDepth.rows; m += cloudDis)
	{
		for (int n = 0; n < imDepth.cols; n += cloudDis)
		{
			float d = imDepth.ptr<float>(m)[n];
			PointT p;
			p.z = d;
			p.x = (n - cx) * p.z / fx;
			p.y = (m - cy) * p.z / fy;
			p.r = 177; //rgb_img.at<cv::Vec3b>(m,n)[2];//177;
			p.g = 177; //rgb_img.at<cv::Vec3b>(m,n)[1];//177;
			p.b = 177; //rgb_img.at<cv::Vec3b>(m,n)[0];//177;
			rgbd_cloud.points.push_back(p);
		}
	}
	rgbd_cloud.height = ceil(imDepth.rows / float(cloudDis));
	rgbd_cloud.width = ceil(imDepth.cols / float(cloudDis));
	// std::cout << "rgbd_cloud: " << rgbd_cloud.points.size() << std::endl;
	// pcl::io::savePCDFileASCII ("plane_depth.pcd", rgbd_cloud);
	return true;
}

void PlaneDetection::ComputePlanesFromOrganizedPointCloud()
{
	int min_plane = 1000; //MultiPlane_SizeMin;//300;
	float AngTh = 3.0;	  // MultiPlane_AngleThre;//2.0;
	float DisTh = 0.5;	  // MultiPlane_DistThre;//0.02;
	// std::cout << "min_plane " << min_plane << std::endl;
	// std::cout << "AngTh " << AngTh << std::endl;
	// std::cout << "DisTh " << DisTh << std::endl;
	// // firstly, compute normal
	// pcl::PointCloud<PointT>::Ptr  input_cloud=rgbd_cloud.makeShared();
	pcl::PointCloud<PointT>::Ptr input_cloud = rgbd_cloud.makeShared();
	pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.05f); // 0.05
	ne.setNormalSmoothingSize(10.0f);  // 10.0
	ne.setInputCloud(input_cloud);
	ne.compute(*cloud_normals);

	// secondly, compute region, label, coefficient, inliners, ...
	pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
	pcl::PointCloud<pcl::Label>::Ptr labels(new pcl::PointCloud<pcl::Label>);
	vector<pcl::ModelCoefficients> coefficients;
	vector<pcl::PointIndices> inliers;
	vector<pcl::PointIndices> label_indices;
	vector<pcl::PointIndices> boundary;
	std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>> regions;
	mps.setMinInliers(min_plane);			   //int min_plane = 1000;
	mps.setAngularThreshold(0.017453 * AngTh); //float AngleThreshold = 3.0 (0.017453=pi/180)
	mps.setDistanceThreshold(DisTh);		   // float DistanceThreshold = 0.05
	mps.setInputNormals(cloud_normals);
	mps.setInputCloud(input_cloud);
	mps.segmentAndRefine(regions, coefficients, inliers, labels, label_indices, boundary);

	// // thirdly, exact and filter point cloud
	for (int i = 0; i < inliers.size(); ++i)
	{
		cv::Mat coef = (cv::Mat_<float>(4, 1) << coefficients[i].values[0],
						coefficients[i].values[1],
						coefficients[i].values[2],
						coefficients[i].values[3]);
		if (coef.at<float>(3) < 0)
			coef = -coef;
		std::cout << "plane: " << i << coef.t() << std::endl;

		pcl::ExtractIndices<PointT> extract;
		extract.setInputCloud(input_cloud);
		extract.setNegative(false);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr planeCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		extract.setIndices(boost::make_shared<pcl::PointIndices>(inliers[i])); // #include <boost/make_shared.hpp>
		extract.filter(*planeCloud);

		// add color
		Eigen::Matrix<double, 6, 3> color_list;
		color_list << 0, 0, 177, 0, 177, 0, 177, 0, 0,
			0, 177, 177, 177, 177, 0, 177, 0, 177;
		for (size_t pt_id = 0; pt_id < planeCloud->size(); pt_id++)
		{
			planeCloud->points[pt_id].r = color_list(i, 0);
			planeCloud->points[pt_id].g = color_list(i, 1);
			planeCloud->points[pt_id].b = color_list(i, 2);
		}

		bool merge_pcl = false;
		for (int j = 0; j < mvPlaneCoefficients.size(); ++j)
		{
			cv::Mat pM = mvPlaneCoefficients[j];
			float d = pM.at<float>(3, 0) - coef.at<float>(3, 0);
			float angle = pM.at<float>(0, 0) * coef.at<float>(0, 0) +
						  pM.at<float>(1, 0) * coef.at<float>(1, 0) +
						  pM.at<float>(2, 0) * coef.at<float>(2, 0);
			if ((d < 0.2 && d > -0.2) && (angle > 0.965 || angle < -0.965)) // associate plane
			{
				// model 1: SACSegmentation // simplest, only single plane
				pcl::PointCloud<PointT> old_points = mvPlanePoints[j];
				for (auto &p : old_points.points)
				{
					p.r = color_list(i, 0);
					p.g = color_list(i, 1);
					p.b = color_list(i, 2);
				}
				pcl::PointCloud<PointT>::Ptr new_cloud = old_points.makeShared();
				*planeCloud += *new_cloud;
				pcl::SACSegmentation<PointT> *seg = new pcl::SACSegmentation<PointT>();
				pcl::ModelCoefficients coefficients;
				pcl::PointIndices inliers;
				seg->setOptimizeCoefficients(true);		// optional
				seg->setModelType(pcl::SACMODEL_PLANE); // required
				seg->setMethodType(pcl::SAC_RANSAC);	// required
				seg->setDistanceThreshold(0.01);		// required 0.01m
				seg->setInputCloud(planeCloud);			// required ptr or setInputCloud(cloud.makeShared());
				seg->segment(inliers, coefficients);	// seg.segment(*inliers, *coefficients);
				if (inliers.indices.size() == 0)
				{
					PCL_ERROR("Could not estimate a planar model for the given dataset.");
					// return false;
				}
				Eigen::Vector4f local_fitted_plane(coefficients.values[0], coefficients.values[1], coefficients.values[2], coefficients.values[3]);
				std::cout << "local_fitted_plane new: " << local_fitted_plane.transpose() << std::endl;
				cv::Mat coef_new = (cv::Mat_<float>(4, 1) << coefficients.values[0],
									coefficients.values[1],
									coefficients.values[2],
									coefficients.values[3]);
				if (coef_new.at<float>(3) < 0)
					coef_new = -coef_new;
				mvPlanePoints[j] = *planeCloud;
				mvPlaneCoefficients[j] = coef_new;
				merge_pcl = true;
				PointCloud::Ptr boundaryPoints(new PointCloud());
				boundaryPoints->points = regions[i].getContour();
				mvBoundaryPoints[j] += *boundaryPoints;
				break;
			}
		}
		if (merge_pcl == false)
		{
			mvPlanePoints.push_back(*planeCloud);
			mvPlaneCoefficients.push_back(coef);
			PointCloud::Ptr boundaryPoints(new PointCloud());
			boundaryPoints->points = regions[i].getContour();
			mvBoundaryPoints.push_back(*boundaryPoints);
		}
	}
	std::cout << "plane num: " << mvPlaneCoefficients.size() << std::endl;
}
