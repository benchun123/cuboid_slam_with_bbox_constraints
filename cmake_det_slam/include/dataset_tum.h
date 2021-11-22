// std c
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;

class dataset_tum
{
public:
    /* data */
    std::vector<std::string> truth_class;
    Eigen::MatrixXd truth_cuboid_list; 

public:
    dataset_tum(/* args */);
    ~dataset_tum();
    void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<int> &vTimestamps);
    void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
        std::vector<std::string> &vstrImageFilenamesD, std::vector<int> &vIndex);
    void LoadObjectList(const string &obj_dim_file);
};

dataset_tum::dataset_tum(/* args */)
{
}

dataset_tum::~dataset_tum()
{
}

