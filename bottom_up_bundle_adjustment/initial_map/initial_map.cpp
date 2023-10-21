#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <iostream>
#include <Eigen/Core>
using namespace cv;
using namespace std;
void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t);

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points
);

inline cv::Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// 像素坐标转相机归一化坐标
Point2f pixel2cam(const Point2d &p, const Mat &K);

void DrawMap(const vector<Point3d> &coordinate_3d);

int main(int argc, char **argv){
    cv::Mat before_img = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat after_img = cv::imread(argv[2], cv::IMREAD_COLOR);
    if(before_img.empty() || after_img.empty()){
        std::cout << "img load fail \n";
        return 1;
    }

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(before_img, after_img, keypoints_1, keypoints_2, matches);

    Mat R, t;keypoints_1, keypoints_2, matches, R, t, points
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);


    Mat K = (Mat_<double>(3, 3) << 322.282, 0, 320.818, 0, 322.282, 178.779, 0, 0, 1);

    Mat img1_plot = before_img.clone();
    Mat img2_plot = after_img.clone();
    for (int i = 0; i < matches.size(); i++) {
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 30, get_color(depth1), 20);

        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 30, get_color(depth2), 20);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

    DrawMap(points);
    
}

void DrawMap(const vector<Point3d> &coordinate_3d) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Draw map", 640, 480);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
      pangolin::ModelViewLookAt(-3,-3,-3, 0,0,0, pangolin::AxisY)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));


  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glLineWidth(5);
    for (size_t i = 0; i < coordinate_3d.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES);
      auto p1 = coordinate_3d[i], p2 = coordinate_3d[i + 1];
      cout << p1.x  << " " << p1.y << " " << p1.z << " i : " << i << "\n";
      glVertex3d(p1.x, p1.y, p1.z);
      glVertex3d(p2.x, p2.y, p2.z);
    }

    glEnd();
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}



void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  vector<DMatch> match;
  matcher->match(descriptors_1, descriptors_2, match);

  double min_dist = 10000, max_dist = 0;

  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t) {
  Mat K = (Mat_<double>(3, 3) << 322.282, 0, 320.818, 0, 322.282, 178.779, 0, 0, 1);

  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  Point2d principal_point(325.1, 249.7);        
  int focal_length = 521;            
  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

  recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points) {
  Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
  Mat T2 = (Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );

  Mat K = (Mat_<double>(3, 3) << 322.282, 0, 320.818, 0, 322.282, 178.779, 0, 0, 1);
  vector<Point2f> pts_1, pts_2;
  for (DMatch m:matches) {
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }

  Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
  //cout << pts_4d.size() << matches.size();

  for (int i = 0; i < pts_4d.cols; i++) {
    Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0); // 归一化
    Point3d p(
      x.at<float>(0, 0),
      x.at<float>(1, 0),
      x.at<float>(2, 0)
    );
    points.push_back(p);
  }
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

