#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, char **argv){
    cv::Mat before_img = cv::imread("../../test_img/after.jpg", cv::IMREAD_COLOR);
    cv::Mat after_img = cv::imread("../../test_img/after.jpg", cv::IMREAD_COLOR);
    if(before_img.empty() || after_img.empty()){
        std::cout << "img load fail \n";
        return 1;
    }

    resize(before_img, before_img, cv::Size(before_img.cols/3, before_img.rows/3));
    resize(after_img, after_img, cv::Size(after_img.cols/3, after_img.rows/3));

    cv::imshow("before", before_img);
    cv::imshow("after", after_img);
    cv::waitKey();
}