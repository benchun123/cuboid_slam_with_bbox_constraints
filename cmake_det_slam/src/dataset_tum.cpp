#include "dataset_tum.h"
#include "detect_cuboid_bbox/matrix_utils.h"




// load association file
void dataset_tum::LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
        std::vector<std::string> &vstrImageFilenamesD, std::vector<int> &vIndex)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        std::string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double id;
            std::string sRGB, sD;
            ss >> id;
            vIndex.push_back(int(id));
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);            
            ss >> id;
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
        }
    }
}

// load rgb or depth file
void dataset_tum::LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<int> &vTimestamps)
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

void dataset_tum::LoadObjectList(const string &obj_dim_file)
{
    // std::vector<std::string> truth_class; 
    // Eigen::MatrixXd obj_dim_list(1,4); 
    truth_cuboid_list.resize(6,9);
    truth_cuboid_list.setZero();
    if (!read_obj_detection_txt(obj_dim_file, truth_cuboid_list, truth_class))
        std::cout << "file not correct: " << obj_dim_file << std::endl;  
    std::cout << "truth_cuboid_list: \n" << truth_cuboid_list << std::endl;  
}
