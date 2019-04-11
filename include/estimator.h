#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <iostream>
#include "multirotor_sim/estimator_base.h"

#include "landing_sim/utils.h"

class Estimator : public multirotor_sim::EstimatorBase
{
private:
public:
  Estimator(std::string filename)
  {
    get_yaml_node("draw_feature_img", filename, draw_feats_);
  }

  virtual ~Estimator()
  {
  }

  // t - current time (seconds)
  // z - imu measurement [acc, gyro]
  // R - imu covariance
  void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R)
  {
    // std::cout << "imu callback" << std::endl;
  }

  void altCallback(const double& t, const Vector1d& z, const Matrix1d& R)
  {
  }
  void mocapCallback(const double& t, const Xformd& z, const Matrix6d& R)
  {
  }

  void simpleCamCallback(const double& t, const ImageFeat& z,
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
  void gnssCallback(const double& t, const Vector6d& z, const Matrix6d& R)
  {
  }

  bool draw_feats_;
};

#endif /* ESTIMATOR_H */
