#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

using namespace Eigen;

void drawImageFeatures(
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>& pixs)
{
  const int radius = 3;
  const cv::Scalar color(255);
  const int thickness = -1; // fill in circle
  cv::Mat img = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);

  for (Vector2d pix : pixs)
  {
    cv::Point2d pt(pix.x(), pix.y());

    cv::circle(img, pt, radius, color, thickness);
  }

  cv::imshow("img", img);
  cv::waitKey(1);
}
