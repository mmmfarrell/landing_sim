#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <iostream>
#include "multirotor_sim/estimator_base.h"

using namespace Eigen;
using namespace multirotor_sim;

class Estimator : public multirotor_sim::EstimatorBase
{
public:
  enum
  {
    xPOS = 0,
    xRHO = 2,
    xVEL = 3,
    xATT = 5,
    xOMEGA = 6,
    xLM = 7,
    xZ = 19
  };

  enum
  {
    SIZE = 4
  };

  typedef Eigen::Matrix<double, xZ, 1> StateVec;
  typedef Eigen::Matrix<double, xZ, xZ> StateMat;
  typedef Eigen::Matrix<double, SIZE, 3> LandmarkVec;

  Estimator(std::string filename);
  virtual ~Estimator();

  // t - current time (seconds)
  // z - imu measurement [acc, gyro]
  // R - imu covariance
  void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R) {}
  void altCallback(const double& t, const Vector1d& z, const Matrix1d& R) {}
  void mocapCallback(const double& t, const Xformd& z, const Matrix6d& R);

  void simpleCamCallback(const double& t, const ImageFeat& z,
                         const Matrix2d& R_pix, const Matrix1d& R_depth);

  // t - current time (seconds)
  // z - gnss measurement [p_{b/ECEF}^ECEF, v_{b/ECEF}^ECEF]
  // R - gnss covariance
  void gnssCallback(const double& t, const Vector6d& z, const Matrix6d& R) {}

  // Kalman Filter
  void propagate(const double& dt);

  void updateGoal(const Vector2d& goal_pix);
  void updateGoalDepth(const double& goal_depth);
  void updateLandmark(const int& id, const Vector2d& lm_pix);

  bool draw_feats_;
  StateVec xhat_;
  StateMat P_;

  double last_time_;
};

#endif /* ESTIMATOR_H */
