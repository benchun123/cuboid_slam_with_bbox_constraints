cmake_minimum_required (VERSION 3.1)

project (sun_rgbd_project)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fpermissive")
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_BUILD_TYPE "Release")

# put FindGLFW.cmake in the source directory of CMakeLists.txt
find_package(OpenCV 3.3 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

include_directories(
  ${PROJECT_SOURCE_DIR}
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/Thirdparty/line_lbd/include
  ${PROJECT_SOURCE_DIR}/Thirdparty/tictoc_profiler/include
  /usr/include/eigen3
  ${PCL_INCLUDE_DIRS}
)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(cuboid_bbox SHARED
    src/detect_cuboid_bbox.cpp
    src/matrix_utils.cpp
    src/object_3d_util.cpp
   )

target_link_libraries(cuboid_bbox
    ${OpenCV_LIBS}
    ${EIGEN3_LIBRARIES} ${Boost_SYSTEM_LIBRARY}
    ${PROJECT_SOURCE_DIR}/Thirdparty/line_lbd/lib/libline_lbd_lib.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/tictoc_profiler/lib/libtictoc_profiler.so
  )

add_library(plane_det SHARED
  src/plane_detection.cpp
 )

target_link_libraries(plane_det
  ${OpenCV_LIBS}
  ${EIGEN3_LIBRARIES} ${Boost_SYSTEM_LIBRARY}
  ${PCL_LIBRARIES}
  )

add_library(dataset_lib SHARED
  src/dataset_tum.cpp
 )

target_link_libraries(dataset_lib
  ${EIGEN3_LIBRARIES} 
  )


## Declare a C++ executable
add_executable(main_rgbd_opti_tum_node src/main_rgbd_opti_tum.cpp)
target_link_libraries(main_rgbd_opti_tum_node
    ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY}
    ${PROJECT_SOURCE_DIR}/Thirdparty/line_lbd/lib/libline_lbd_lib.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/tictoc_profiler/lib/libtictoc_profiler.so
    cuboid_bbox
    plane_det
    dataset_lib
    ) 

