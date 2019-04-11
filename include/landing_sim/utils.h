#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

void drawImageFeatures(
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>& pixs)
{
  //std::cout << "hello" << std::endl;

  Eigen::Vector2d pix = pixs[0];

  cv::Point2d pt(pix.x(), pix.y());

  cv::Mat img = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
  const int radius = 3;
  const cv::Scalar color(255);
  const int thickness = -1; // fill in circle
  cv::circle(img, pt, radius, color, thickness);
  cv::imshow("img", img);
  cv::waitKey(1);
}
