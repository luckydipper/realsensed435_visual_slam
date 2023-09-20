cmake_minimum_required(VERSION 3.1)

project(test_two_img_randering)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_two_img_randering.out test_two_img_randering.cpp)
target_link_libraries(test_two_img_randering.out ${OpenCV_LIBS})