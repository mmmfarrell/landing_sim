#include <iostream>

#include "landing_sim/utils.h"
#include "multirotor_sim/utils.h"

#include "estimator.h"

using namespace Eigen;
using namespace multirotor_sim;

Estimator::Estimator(std::string filename)
{
  get_yaml_node("draw_feature_img", filename, draw_feats_);
  get_yaml_eigen("x0", filename, xhat_);

  StateVec P_diag;
  get_yaml_eigen("P0", filename, P_diag);
  P_ = P_diag.asDiagonal();
}

Estimator::~Estimator()
{
}

// t - current time (seconds)
// z - imu measurement [acc, gyro]
// R - imu covariance
void Estimator::imuCallback(const double& t, const Vector6d& z, const Matrix6d& R)
{
  // std::cout << "imu callback" << std::endl;
}

void Estimator::altCallback(const double& t, const Vector1d& z, const Matrix1d& R)
{
}
void Estimator::mocapCallback(const double& t, const Xformd& z, const Matrix6d& R)
{
}

void Estimator::simpleCamCallback(const double& t, const ImageFeat& z,
                       const Matrix2d& R_pix, const Matrix1d& R_depth)
{
  // std::cout << "simple cam callback" << std::endl;
  if (draw_feats_)
    drawImageFeatures(z.pixs);
  // std::cout << "pix: " << z.pixs[0] << std::endl;
  // std::vector<Vector2d, aligned_allocator<Vector2d>> pixs; // pixel
  // measurements in this image
}

// t - current time (seconds)
// z - gnss measurement [p_{b/ECEF}^ECEF, v_{b/ECEF}^ECEF]
// R - gnss covariance
void Estimator::gnssCallback(const double& t, const Vector6d& z, const Matrix6d& R)
{
}

