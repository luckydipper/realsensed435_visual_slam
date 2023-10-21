cmake_minimum_required(VERSION 3.1)

project(FAST_feature_matching)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
#set(OPENCV_ENABLE_NONFREE = ON)
#-D OPENCV_ENABLE_NONFREE = ON
add_executable(FAST_feature_matching.out FAST_feature_matching.cpp)
target_link_libraries(FAST_feature_matching.out ${OpenCV_LIBS})