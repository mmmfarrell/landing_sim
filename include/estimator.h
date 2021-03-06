#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <iostream>
#include "multirotor_sim/estimator_base.h"
#include "geometry/quat.h"

#define PRINTMAT(x) std::cout << #x << std::endl << x << std::endl;

using namespace Eigen;
using namespace multirotor_sim;

class Estimator : public multirotor_sim::EstimatorBase
{
public:
  enum
  {
    xUAVATT = 0,
    xPOS = 3,
    xRHO = 5,
    xVEL = 6,
    xATT = 8,
    xOMEGA = 9,
    xLM = 10,
    xZ = 22
  };

  enum
  {
    uOMEGA = 0,
    uVEL = 3,
    uZ = 6
  };

  enum
  {
    SIZE = 4
  };

  typedef Eigen::Matrix<double, xZ, 1> StateVec;
  typedef Eigen::Matrix<double, xZ, xZ> StateMat;
  typedef Eigen::Matrix<double, SIZE, 3> LandmarkVec;
  typedef Eigen::Matrix<double, uZ, 1> InputVec;

  Estimator(std::string filename);
  virtual ~Estimator();

  // t - current time (seconds)
  // z - imu measurement [acc, gyro]
  // R - imu covariance
  void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R);
  void altCallback(const double& t, const Vector1d& z, const Matrix1d& R)
  {
  }
  void mocapCallback(const double& t, const Xformd& z, const Matrix6d& R);
  void velocityCallback(const double& t, const Vector3d& vel_b,
                        const Matrix3d& R);

  void simpleCamCallback(const double& t, const ImageFeat& z,
                         const Matrix2d& R_pix, const Matrix1d& R_depth);

  // t - current time (seconds)
  // z - gnss measurement [p_{b/ECEF}^ECEF, v_{b/ECEF}^ECEF]
  // R - gnss covariance
  void gnssCallback(const double& t, const Vector6d& z, const Matrix6d& R)
  {
  }

  void initLandmarks(
      const std::vector<Eigen::Vector2d,
                        Eigen::aligned_allocator<Eigen::Vector2d>>& pixs);
  void setInverseDepth();
  void toRot(const double theta, Matrix2d& R);
  Matrix2d dtheta_R(const double theta);

  // Kalman Filter
  void propagate(const double& time_step, const InputVec& u_in);

  void updateUAVAttitude(const Vector3d& uav_euler);
  void updateGoal(const Vector2d& goal_pix);
  void updateGoalDepth(const double& goal_depth);
  void updateLandmark(const int& id, const Vector2d& lm_pix);
  void updateLandmarkDepth(const int& id, const double& lm_depth);

  Vector2d invserseMeasurementModelLandmark(const Vector2d& lm_pix,
                                            const double rho);
  Vector2d virtualImagePixels(const Vector2d& pix);

  Matrix3d R_I_b() const;

  // Kalman filter
  StateVec xhat_;
  StateMat P_;

  // Prop
  InputVec u_in_;
  StateVec xdot_;
  StateMat A_;
  StateMat Q_;

  // Update
  const StateMat I_ = StateMat::Identity();
  Vector3d att_residual_;
  Matrix3d att_R_;
  Eigen::Matrix<double, 3, xZ> att_H_;
  Eigen::Matrix<double, xZ, 3> att_K_;

  Vector2d pix_residual_;
  Matrix2d pix_R_;
  Eigen::Matrix<double, 2, xZ> pix_H_;
  Eigen::Matrix<double, xZ, 2> pix_K_;

  Matrix1d depth_R_;
  Eigen::Matrix<double, 1, xZ> depth_H_;

  // Camera params
  double fx_;
  double fy_;
  double cx_;
  double cy_;
  quat::Quatd q_b_c_;
  double min_depth_;

  // UAV stuff
  quat::Quatd q_I_b_;

  int num_prop_steps_;
  double last_time_;
  bool feats_initialized_;
  bool draw_feats_;
  bool record_vid_;
  bool update_goal_depth_;

  double use_goal_stop_time_;
};

#endif /* ESTIMATOR_H */
