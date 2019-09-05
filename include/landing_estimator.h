#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <iostream>

#include <list>
#include "multirotor_sim/estimator_base.h"
#include "geometry/quat.h"

#define PRINTMAT(x) std::cout << #x << std::endl << x << std::endl;
#define GRAVITY 9.81

using namespace Eigen;
using namespace multirotor_sim;

class Estimator : public multirotor_sim::EstimatorBase
{
public:
  enum
  {
    xPOS = 0,
    xATT = 3,
    xVEL = 6,
    xMU = 9,
    xBA = 10,
    xBW = 13,
    xGOAL_POS = 16,
    xGOAL_RHO = 18,
    xGOAL_VEL = 19,
    xGOAL_ATT = 21,
    xGOAL_OMEGA = 22,
    xGOAL_LM = 23,
     xZ = 35  // xGOAL_LM + 3 * MAXLANDMARKs
    //xZ = 83  // xGOAL_LM + 3 * MAXLANDMARKs
  };

  enum
  {
    uAZ = 0,
    uOMEGA = 1,
    uZ = 4
  };

  enum
  {
     MAXLANDMARKS = 4
    //MAXLANDMARKS = 20
  };

  enum
  {
    zDIMS = 3
  };

  typedef Eigen::Matrix<double, xZ, 1> StateVec;
  typedef Eigen::Matrix<double, xZ, xZ> StateMat;
  typedef Eigen::Matrix<double, uZ, 1> InputVec;
  typedef Eigen::Matrix<double, uZ, uZ> InputMat;
  typedef Eigen::Matrix<double, xZ, uZ> StateInputMat;

  typedef Eigen::Matrix<double, zDIMS, 1> MeasVec;
  typedef Eigen::Matrix<double, zDIMS, zDIMS> MeasMat;
  typedef Eigen::Matrix<double, zDIMS, xZ> MeasH;
  typedef Eigen::Matrix<double, xZ, zDIMS> MeasK;

  Estimator(std::string filename);
  virtual ~Estimator();

  // t - current time (seconds)
  // z - imu measurement [acc, gyro]
  // R - imu covariance
  void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R);
  void altCallback(const double& t, const Vector1d& z, const Matrix1d& R);
  void mocapCallback(const double& t, const Xformd& z, const Matrix6d& R);
  void velocityCallback(const double& t, const Vector3d& vel_b,
                        const Matrix3d& R);

  void arucoCallback(const double& t, const Vector2d& pix, const double& depth,
                     const Matrix2d& R_pix, const Matrix1d& R_depth);
  void landmarksCallback(const double& t, const ImageFeat& z,
                         const Matrix2d& R_pix);

  // t - current time (seconds)
  // z - gnss measurement [p_{b/I}^I v_{b/I}^I]
  // R - gnss covariance
  void gnssCallback(const double& t, const Vector6d& z, const Matrix6d& R);

  void propagate(const double& dt, const InputVec& u_in);
  void update(const double dims, const MeasVec& residual, const MeasMat& R,
              const MeasH& H);
  void dynamics(const StateVec& x, const InputVec& u_in, StateVec& xdot,
                StateMat& A, StateInputMat& G);

  void printLmIDs();
  void printLmXhat();
  void printLmPhat();
  void initLandmark(const int& id, const Vector2d& pix);
  void removeLandmark(const int& idx, const int& lm_id);
  void updateLandmark(const int& idx, const Vector2d& pix,
                      const Matrix2d& R_pix);

  // EKF Member variables
  StateVec xhat_;
  StateMat P_;
  StateMat Qx_;
  InputMat Qu_;

  InputVec u_;
  StateVec xdot_;
  StateMat A_;
  StateInputMat G_;

  MeasVec z_resid_;
  MeasMat z_R_;
  MeasH H_;
  MeasK K_;

  StateVec lambda_;
  StateMat lambda_mat_;

  // Landmark Initialization stuff
  Vector3d x0_landmarks_;
  Vector3d P0_landmarks_;

  const StateMat I_ = StateMat::Identity();

  double last_prop_time_;
  double use_goal_stop_time_;
  bool use_partial_update_;

  std::list<int> landmark_ids_;
};

#endif /* ESTIMATOR_H */
