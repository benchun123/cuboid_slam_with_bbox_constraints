cmake_minimum_required( VERSION 2.8)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( CMAKE_BUILD_TYPE Release) # maybe make no effect, google it

project(bbox_constraints_cmake)
find_package( OpenCV REQUIRED) 
find_package(Boost REQUIRED COMPONENTS system)


include_directories( 
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    "/usr/include/eigen3/"
    ${OpenCV_INCLUDE_DIRS} 
    ${PROJECT_SOURCE_DIR}/Thirdparty/tictoc_profiler/include
    ${PROJECT_SOURCE_DIR}/Thirdparty/line_lbd/include
)

add_executable(bbox_cabinet_det_node main_bbox_constraints_detection.cpp)
target_link_libraries(bbox_cabinet_det_node
    ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY}
    ${PROJECT_SOURCE_DIR}/Thirdparty/line_lbd/lib/libline_lbd_lib.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/tictoc_profiler/lib/libtictoc_profiler.so
) 

# add_executable(bbox_cabinet_slam_node cabinet_slam_single.cpp)
# target_link_libraries(bbox_cabinet_slam_node
#     ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY}
#     ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
# ) 
