#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>


int main(){
    cv::Mat before_img = cv::imread("../../test_img/after.jpg", cv::IMREAD_COLOR);
    cv::Mat after_img = cv::imread("../../test_img/after.jpg", cv::IMREAD_COLOR);
    if(before_img.empty() || after_img.empty()){
        std::cout << "img load fail \n";
        return 1;
    }
    resize(before_img, before_img, cv::Size(before_img.cols/3, before_img.rows/3));
    resize(after_img, after_img, cv::Size(after_img.cols/3, after_img.rows/3));
  
    std::vector<cv::KeyPoint> before_features, after_features;

    cv::Mat descriptor_bef, descriptor_aft;
    
    cv::FAST(before_img, before_features, 40, false);
    cv::drawKeypoints(before_img,before_features,before_img);
 
    cv::FAST(after_img, after_features, 40, false);
    cv::drawKeypoints(after_img,after_features,after_img);

    cv::imshow("before", before_img);
    cv::imshow("after", after_img);
    cv::waitKey();

}