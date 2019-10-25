#pragma once

#include <Eigen/Core>
#include <vector>

// using namespace Eigen;
using Vector2d = Eigen::Vector2d;
template<typename T> using aligned_allocator = Eigen::aligned_allocator<T>;

class ImageFeat
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id; // image label
  double t; // time stamp of this image
  // std::vector<Vector2d, aligned_allocator<Vector2d>> pixs; // pixel measurements in this image
  std::vector<Vector2d> pixs; // pixel measurements in this image
  std::vector<int> feat_ids; // feature ids corresonding to pixel measurements

  void reserve(const int& N)
  {
    pixs.reserve(N);
    feat_ids.reserve(N);
  }

  void clear()
  {
    pixs.clear();
    feat_ids.clear();
  }
};

