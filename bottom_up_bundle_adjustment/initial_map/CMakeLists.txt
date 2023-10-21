cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_FLAGS "-std=c++14")

project(test_two_img_randering)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

#include_directories()
include_directories(${OpenCV_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS}  "/usr/include/eigen3" )
 
add_executable(initial_map.out initial_map.cpp)
target_link_libraries(initial_map.out ${OpenCV_LIBS} ${Pangolin_LIBRARIES})