#include <iostream>

#include "landing_sim/utils.h"
#include "multirotor_sim/utils.h"

#include "landing_estimator.h"
#include "estimator_utils.h"
#include "measurement_models.h"

using namespace Eigen;
using namespace multirotor_sim;

Estimator::Estimator(std::string filename)
{
  last_prop_time_ = -999.;

  // Load yaml params
  get_yaml_node("use_goal_stop_time", filename, use_goal_stop_time_);
  get_yaml_node("use_partial_update", filename, use_partial_update_);

  // Unified Inverse Depth Parametrization for Monocular SLAM
  // by Montiel, et. al.
  double dmin;
  get_yaml_node("min_goal_depth", filename, dmin);
  const double rho_min = 1. / dmin;
  const double rho_init = rho_min / 2.;
  const double phat_rho_init = rho_min * rho_min / 16;

  // Init EKF State
  Eigen::Matrix<double, xGOAL_LM, 1> x0_states;
  get_yaml_eigen("x0_states", filename, x0_states);
  get_yaml_eigen("x0_landmarks", filename, x0_landmarks_);

  xhat_.block<xGOAL_LM, 1>(0, 0) = x0_states;
  xhat_(xGOAL_RHO) = rho_init;

  Eigen::Matrix<double, xGOAL_LM, 1> P0_states;
  get_yaml_eigen("P0_states", filename, P0_states);
  get_yaml_eigen("P0_landmarks", filename, P0_landmarks_);

  P_.block<xGOAL_LM, xGOAL_LM>(0, 0) = P0_states.asDiagonal();
  P_(xGOAL_RHO, xGOAL_RHO) = phat_rho_init;

  Eigen::Matrix<double, xGOAL_LM, 1> Qx_states;
  Eigen::Vector3d Qx_landmarks;
  get_yaml_eigen("Qx_states", filename, Qx_states);
  get_yaml_eigen("Qx_landmarks", filename, Qx_landmarks);

  Qx_.setZero();
  Qx_.block<xGOAL_LM, xGOAL_LM>(0, 0) = Qx_states.asDiagonal();

  InputVec Qu;
  get_yaml_eigen("Qu", filename, Qu);
  Qu_ = Qu.asDiagonal();

  // Populate the partial update lambda vector and matrix
  Eigen::Matrix<double, xGOAL_LM, 1> lambda_states;
  Eigen::Vector3d lambda_landmarks;
  get_yaml_eigen("lambda_states", filename, lambda_states);
  get_yaml_eigen("lambda_landmarks", filename, lambda_landmarks);

  lambda_.block<xGOAL_LM, 1>(0, 0) = lambda_states;

  for (int i = 0; i < MAXLANDMARKS; ++i)
  {
    int LM_IDX = xGOAL_LM + 3 * i;
    xhat_.block<3, 1>(LM_IDX, 0) = x0_landmarks_;
    P_.block<3, 3>(LM_IDX, LM_IDX) = P0_landmarks_.asDiagonal();
    Qx_.block<3, 3>(LM_IDX, LM_IDX) = Qx_landmarks.asDiagonal();
    lambda_.block<3, 1>(LM_IDX, 0) = lambda_landmarks;
  }

  const StateVec ones = StateVec::Constant(1.0);
  lambda_mat_ = ones * lambda_.transpose() + lambda_ * ones.transpose() -
                lambda_ * lambda_.transpose();

  // Camera Parameters
  get_yaml_eigen("cam_K", filename, cam_K_);
  cam_K_inv_ = cam_K_.inverse();
  get_yaml_eigen("p_b_c", filename, p_b_c_);
  get_yaml_eigen("q_b_c", filename, q_b_c_.arr_);
  q_b_c_.normalize();
}

Estimator::~Estimator()
{
}

void Estimator::imuCallback(const double& t, const Vector6d& z,
                            const Matrix6d& R)
{
  u_.setZero();
  u_(uAZ) = z(2);
  u_.segment<3>(uOMEGA) = z.segment<3>(3);

  double dt = t - last_prop_time_;
  if ((dt < 0.1) && (dt > 0))
  {
    this->propagate(dt, u_);
  }
  last_prop_time_ = t;

  // update with ax, ay
  const double imu_dims = 2;
  const double xhat_mu = xhat_(xMU);
  const Eigen::Vector2d xhat_accel =
      -xhat_mu * xhat_.segment<2>(xVEL) + xhat_.segment<2>(xBA);
  z_resid_.head(imu_dims) = z.head(2) - xhat_accel;
  z_R_.topLeftCorner(imu_dims, imu_dims) = R.topLeftCorner(imu_dims, imu_dims);

  H_.setZero();
  H_(0, xVEL + 0) = -xhat_mu;
  H_(0, xMU) = -xhat_(xVEL + 0);
  H_(0, xBA + 0) = 1.;
  H_(1, xVEL + 1) = -xhat_mu;
  H_(1, xMU) = -xhat_(xVEL + 1);
  H_(1, xBA + 1) = 1.;

  update(imu_dims, z_resid_, z_R_, H_);
}

void Estimator::altCallback(const double& t, const Vector1d& z,
                            const Matrix1d& R)
{
  const double alt_dims = 1;
  z_resid_.head(alt_dims) = z + xhat_.segment<1>(xPOS + 2);
  z_R_.topLeftCorner(alt_dims, alt_dims) = R;

  H_.setZero();
  H_.block<1, 1>(0, xPOS + 2) = -1 * Matrix1d::Identity();

  update(alt_dims, z_resid_, z_R_, H_);
}

void Estimator::mocapCallback(const double& t, const Xformd& z,
                              const Matrix6d& R)
{
  //// update position
  // const double pos_dims = 3;
  // const Vector3d mocap_pos = z.t();
  // z_resid_.head(pos_dims) = mocap_pos - xhat_.segment<3>(xPOS);
  // z_R_.topLeftCorner(pos_dims, pos_dims) = 0.001 *
  // Eigen::Matrix3d::Identity();

  // H_.setZero();
  // H_.block<3, 3>(0, xPOS) = Eigen::Matrix3d::Identity();

  // update(pos_dims, z_resid_, z_R_, H_);

  // Update atttitude
  const double att_dims = 3;
  const Vector3d mocap_euler = z.q().euler();
  z_resid_.head(att_dims) = mocap_euler - xhat_.segment<3>(xATT);
  // z_R_.topLeftCorner(att_dims, att_dims) = 0.001 *
  // Eigen::Matrix3d::Identity();
  z_R_.topLeftCorner(att_dims, att_dims) = R.block<3, 3>(3, 3);

  H_.setZero();
  H_.block<3, 3>(0, xATT) = Eigen::Matrix3d::Identity();

  update(att_dims, z_resid_, z_R_, H_);

  //// Update atttitude no yaw
  // const double att_dims = 2;
  // const Vector3d mocap_euler = z.q().euler();
  // z_resid_.head(att_dims) = mocap_euler.head(2) - xhat_.segment<2>(xATT);
  ////z_R_.topLeftCorner(att_dims, att_dims) = 0.001 *
  /// Eigen::Matrix2d::Identity();
  // z_R_.topLeftCorner(att_dims, att_dims) = R.block<2, 2>(3, 3);

  // H_.setZero();
  // H_.block<2, 2>(0, xATT) = Eigen::Matrix2d::Identity();

  // update(att_dims, z_resid_, z_R_, H_);
}

// void Estimator::velocityCallback(const double& t, const Vector3d& vel_b,
// const Matrix3d& R)
//{
//}

void Estimator::arucoCallback(const double& t, const Vector2d& pix,
                              const double& depth, const Matrix2d& R_pix,
                              const Matrix1d& R_depth)
{
  if (t < use_goal_stop_time_)
  {
    // Update goal depth
    int depth_dims = 0;
    MeasVec depth_zhat;
    goalDepthMeasModel(xhat_, depth_dims, depth_zhat, H_, p_b_c_, q_b_c_);
    z_resid_(0) = depth - depth_zhat(0);
    z_R_.topLeftCorner(depth_dims, depth_dims) = R_depth;
    update(depth_dims, z_resid_, z_R_, H_);

    // Update goal pixel
    int pix_dims = 0;
    MeasVec pix_zhat;
    goalPixelMeasModel(xhat_, pix_dims, pix_zhat, H_, cam_K_, p_b_c_, q_b_c_);
    z_resid_.head(pix_dims) = pix - pix_zhat.head(pix_dims);
    z_R_.topLeftCorner(pix_dims, pix_dims) = R_pix;
    update(pix_dims, z_resid_, z_R_, H_);
  }
}

void Estimator::landmarksCallback(const double& t, const ImageFeat& z,
                                  const Matrix2d& R_pix)
{
  std::list<int>::iterator it = landmark_ids_.begin();

  int idx = 0;
  const int num_landmarks = z.pixs.size();
  while (idx < num_landmarks)
  {
    const int lm_id = z.feat_ids[idx];
    const Eigen::Vector2d lm_pix = z.pixs[idx];
    if (it == landmark_ids_.end())
    {
      initLandmark(lm_id, lm_pix);
      idx++;
    }
    else
    {
      const int expected_lm_id = *it;
      it++;
      if (expected_lm_id != lm_id)
      {
        removeLandmark(idx, expected_lm_id);
      }
      else
      {
        updateLandmark(idx, lm_pix, R_pix);
        idx++;
      }
    }
  }
}

void Estimator::gnssCallback(const double& t, const Vector6d& z,
                             const Matrix6d& R)
{
  // update position
  const double pos_dims = 3;
  const Vector3d gnss_pos = z.head(pos_dims);
  z_resid_.head(pos_dims) = gnss_pos - xhat_.segment<3>(xPOS);
  z_R_.topLeftCorner(pos_dims, pos_dims) = R.topLeftCorner(pos_dims, pos_dims);

  H_.setZero();
  H_.block<3, 3>(0, xPOS) = Eigen::Matrix3d::Identity();

  update(pos_dims, z_resid_, z_R_, H_);

  // update velocity
  // const double vel_dims = 3;
  // const Eigen::Vector3d gnss_vel_I = z.segment<3>(3);

  // const double phi = xhat_(xATT + 0);
  // const double theta = xhat_(xATT + 1);
  // const double psi = xhat_(xATT + 2);
  // const Eigen::Matrix3d R_I_b = rotmItoB(phi, theta, psi);
  // const Eigen::Vector3d xhat_vel_b = xhat_.segment<3>(xVEL);
  // const Eigen::Vector3d zhat_vel_I = R_I_b.transpose() * xhat_vel_b;

  // z_resid_.head(vel_dims) = gnss_vel_I - zhat_vel_I;
  // z_R_.topLeftCorner(vel_dims, vel_dims) = R.block<3, 3>(3, 3);

  // const Eigen::Matrix3d d_R_d_phi = dRIBdPhi(phi, theta, psi);
  // const Eigen::Matrix3d d_R_d_theta = dRIBdTheta(phi, theta, psi);
  // const Eigen::Matrix3d d_R_d_psi = dRIBdPsi(phi, theta, psi);
  // H_.setZero();
  // H_.block<3, 1>(0, xATT + 0) = d_R_d_phi * xhat_vel_b;
  // H_.block<3, 1>(0, xATT + 1) = d_R_d_theta * xhat_vel_b;
  // H_.block<3, 1>(0, xATT + 2) = d_R_d_psi * xhat_vel_b;
  // H_.block<3, 3>(0, xVEL) = R_I_b.transpose();

  // update(vel_dims, z_resid_, z_R_, H_);
}

void Estimator::propagate(const double& dt, const InputVec& u_in)
{
  dynamics(xhat_, u_, xdot_, A_, G_);

  xhat_ += xdot_ * dt;

  const int num_landmarks = landmark_ids_.size();
  const int num_states = xGOAL_LM + 3 * num_landmarks;
  auto Gsmall = G_.topRows(num_states);
  auto Asmall = A_.topLeftCorner(num_states, num_states);
  auto Psmall = P_.topLeftCorner(num_states, num_states);
  auto Qxsmall = Qx_.topLeftCorner(num_states, num_states);
  auto Ismall = I_.topLeftCorner(num_states, num_states);

  Gsmall = (Ismall + Asmall * dt / 2.0 + Asmall * Asmall * dt * dt / 6.0) *
           Gsmall * dt;
  Asmall = Ismall + Asmall * dt + Asmall * Asmall * dt * dt / 2.0;
  Psmall = Asmall * Psmall * Asmall.transpose() +
           Gsmall * Qu_ * Gsmall.transpose() + Qxsmall;
}

void Estimator::update(const double dims, const MeasVec& residual,
                       const MeasMat& R, const MeasH& H)
{
  const int num_landmarks = landmark_ids_.size();
  const int num_states = xGOAL_LM + 3 * num_landmarks;

  // Create small versions of each matrix to only do math on valid states
  // Note this is not necessary. Estimator works great without this, it just
  // speeds it up a bunch when there are consistently less landmarks tracked
  // than MAXLANDMARKS
  auto Ksmall = K_.topRows(num_states);
  auto Hsmall = H.leftCols(num_states);
  auto Psmall = P_.topLeftCorner(num_states, num_states);
  auto Asmall = A_.topLeftCorner(num_states, num_states);
  auto xhatsmall = xhat_.topRows(num_states);
  auto lambdasmall = lambda_.topRows(num_states);
  auto lambdamatsmall = lambda_mat_.topLeftCorner(num_states, num_states);
  auto Ismall = I_.topLeftCorner(num_states, num_states);

  Ksmall.leftCols(dims) =
      Psmall * Hsmall.topRows(dims).transpose() *
      (Hsmall.topRows(dims) * Psmall * Hsmall.topRows(dims).transpose() +
       R.topLeftCorner(dims, dims))
          .inverse();

  if (use_partial_update_)
  {
    // Apply Fixed Gain Partial update per
    // "Partial-Update Schmidt-Kalman Filter" by Brink
    // Modified to operate inline and on the manifold
    xhatsmall +=
        lambdasmall.asDiagonal() * Ksmall.leftCols(dims) * residual.head(dims);
    Asmall = Ismall - Ksmall.leftCols(dims) * Hsmall.topRows(dims);
    Psmall += lambdamatsmall.cwiseProduct(
        Asmall * Psmall * Asmall.transpose() +
        Ksmall.leftCols(dims) * R.topLeftCorner(dims, dims) *
            Ksmall.leftCols(dims).transpose() -
        Psmall);
  }
  else
  {
    xhatsmall += Ksmall.leftCols(dims) * residual.head(dims);
    // just reuse A_ to save memory
    Asmall = Ismall - Ksmall.leftCols(dims) * Hsmall.topRows(dims);
    Psmall = Asmall * Psmall * Asmall.transpose() +
             Ksmall.leftCols(dims) * R.topLeftCorner(dims, dims) *
                 Ksmall.leftCols(dims).transpose();
  }
}

void Estimator::dynamics(const StateVec& x, const InputVec& u_in,
                         StateVec& xdot, StateMat& A, StateInputMat& G)
{
  // UAV States
  const double phi = x(xATT + 0);
  const double theta = x(xATT + 1);
  const double psi = x(xATT + 2);
  const double u = x(xVEL + 0);
  const double v = x(xVEL + 1);
  const double w = x(xVEL + 2);
  const double mu = x(xMU);

  // Landing Goal States
  const double rho_g = x(xGOAL_RHO);
  const Eigen::Vector2d vel_g = x.segment<2>(xGOAL_VEL);
  const double theta_g = x(xGOAL_ATT);
  const double omega_g = x(xGOAL_OMEGA);

  const double az = u_in(uAZ) + x(xBA + 2);
  const double p = u_in(uOMEGA + 0) + x(xBW + 0);
  const double q = u_in(uOMEGA + 1) + x(xBW + 1);
  const double r = u_in(uOMEGA + 2) + x(xBW + 2);

  const Eigen::Vector3d vel_b = x.segment<3>(xVEL);
  // const Eigen::Vector3d pqr = u_in.segment<3>(uOMEGA);
  const Eigen::Vector3d pqr(p, q, r);

  const Eigen::Matrix3d R_I_b = rotmItoB(phi, theta, psi);
  const Eigen::Matrix3d wmat = wMat(phi, theta, psi);

  static const Eigen::Vector3d grav_I(0., 0., GRAVITY);
  const Eigen::Vector3d grav_b = R_I_b * grav_I;

  static const Eigen::Matrix<double, 2, 3> I_2x3 = I_.block<2, 3>(0, 0);
  static const Eigen::Vector3d e3(0., 0., 1.);

  // Dynamics
  xdot.setZero();

  // UAV Dynamics
  xdot.segment<3>(xPOS) = R_I_b.transpose() * vel_b;
  xdot.segment<3>(xATT) = wmat * pqr;
  xdot(xVEL + 0) = grav_b(0) + v * r - w * q - mu * u;
  xdot(xVEL + 1) = grav_b(1) + w * p - u * r - mu * v;
  xdot(xVEL + 2) = grav_b(2) + u * q - v * p + az;
  xdot(xMU) = 0.;

  // Jacobian
  A.setZero();
  const double cp = cos(phi);
  const double sp = sin(phi);
  const double ct = cos(theta);
  const double st = sin(theta);
  const double tt = tan(theta);
  const double cpsi = cos(psi);
  const double spsi = sin(psi);

  const Eigen::Matrix3d d_R_d_phi = dRIBdPhi(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_theta = dRIBdTheta(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_psi = dRIBdPsi(phi, theta, psi);

  // Pos dot
  // d pdot / d att
  // A(xPOS + 0, xATT + 0) =
  //(cp * st * cpsi + sp * spsi) * v + (-sp * st * cpsi + cp * spsi) * w;
  // A(xPOS + 0, xATT + 1) =
  //(-st * cpsi) * u + (sp * ct * cpsi) * v + (cp * ct * cpsi) * w;
  // A(xPOS + 0, xATT + 2) = (-ct * spsi) * u + (-sp * st * spsi - cp * cpsi) *
  // v +
  //(-cp * st * spsi + sp * cpsi) * w;

  // A(xPOS + 1, xATT + 0) =
  //(cp * st * spsi - sp * cpsi) * v + (-sp * st * spsi - cp * cpsi) * w;
  // A(xPOS + 1, xATT + 1) =
  //(-st * spsi) * u + (sp * ct * spsi) * v + (cp * ct * spsi) * w;
  // A(xPOS + 1, xATT + 2) = (ct * cpsi) * u + (sp * st * cpsi - cp * spsi) * v
  // + (cp * st * cpsi + sp * spsi) * w;

  // A(xPOS + 2, xATT + 0) = (cp * ct) * v + (-sp * ct) * w;
  // A(xPOS + 2, xATT + 1) = (-ct) * u + (-sp * st) * v + (-cp * st) * w;
  // A(xPOS + 2, xATT + 2) = 0.;
  A.block<3, 1>(xPOS, xATT + 0) = d_R_d_phi.transpose() * vel_b;
  A.block<3, 1>(xPOS, xATT + 1) = d_R_d_theta.transpose() * vel_b;
  A.block<3, 1>(xPOS, xATT + 2) = d_R_d_psi.transpose() * vel_b;

  // d pdot / d vel
  A.block<3, 3>(xPOS, xVEL) = R_I_b.transpose();

  // Att dot
  // d attdot / d att
  A(xATT + 0, xATT + 0) = cp * tt * q - sp * tt * r;
  A(xATT + 0, xATT + 1) = (sp / (ct * ct)) * q + (cp / (ct * ct)) * r;
  A(xATT + 0, xATT + 2) = 0.;

  A(xATT + 1, xATT + 0) = -sp * q - cp * r;
  A(xATT + 1, xATT + 1) = 0.;
  A(xATT + 1, xATT + 2) = 0.;

  A(xATT + 2, xATT + 0) = (cp / ct) * q + (-sp / ct) * r;
  A(xATT + 2, xATT + 1) = (sp / ct) * tt * q + (cp / ct) * tt * r;
  A(xATT + 2, xATT + 2) = 0.;

  A.block<3, 3>(xATT, xBW) = -wmat;

  // Vel dot
  // d u dot / d everything
  A(xVEL + 0, xATT + 0) = 0.;
  A(xVEL + 0, xATT + 1) = -ct * GRAVITY;
  A(xVEL + 0, xATT + 2) = 0.;
  A(xVEL + 0, xVEL + 0) = -mu;
  A(xVEL + 0, xVEL + 1) = r;
  A(xVEL + 0, xVEL + 2) = -q;
  A(xVEL + 0, xMU) = -u;

  // d v dot / d everything
  A(xVEL + 1, xATT + 0) = cp * ct * GRAVITY;
  A(xVEL + 1, xATT + 1) = -sp * st * GRAVITY;
  A(xVEL + 1, xATT + 2) = 0.;
  A(xVEL + 1, xVEL + 0) = -r;
  A(xVEL + 1, xVEL + 1) = -mu;
  A(xVEL + 1, xVEL + 2) = p;
  A(xVEL + 1, xMU) = -v;

  // d w dot / d everything
  A(xVEL + 2, xATT + 0) = -sp * ct * GRAVITY;
  A(xVEL + 2, xATT + 1) = -cp * st * GRAVITY;
  A(xVEL + 2, xATT + 2) = 0.;
  A(xVEL + 2, xVEL + 0) = q;
  A(xVEL + 2, xVEL + 1) = -p;
  A(xVEL + 2, xVEL + 2) = 0.;
  A(xVEL + 2, xMU) = 0.;
  A(xVEL + 2, xBA + 2) = 1.;

  A.block<3, 3>(xVEL, xBW) = -skewMat(vel_b);

  // Input Jacobian
  G.setZero();

  // attitude dot
  G.block<3, 3>(xATT, uOMEGA) = wmat;

  // velocity dot
  G(xVEL + 0, uAZ) = 0.;
  G(xVEL + 0, uOMEGA + 0) = 0.;
  G(xVEL + 0, uOMEGA + 1) = -w;
  G(xVEL + 0, uOMEGA + 2) = v;

  G(xVEL + 1, uAZ) = 0.;
  G(xVEL + 1, uOMEGA + 0) = w;
  G(xVEL + 1, uOMEGA + 1) = 0.;
  G(xVEL + 1, uOMEGA + 2) = -u;

  G(xVEL + 2, uAZ) = -1.;
  G(xVEL + 2, uOMEGA + 0) = -v;
  G(xVEL + 2, uOMEGA + 1) = u;
  G(xVEL + 2, uOMEGA + 2) = 0.;

  // Landing goal
  const Eigen::Matrix2d R_v_g = rotm2dItoB(theta_g);
  const Eigen::Matrix2d dR_v_g_dTheta = dR2DdTheta(theta_g);

  // Landing goal dynamics
  xdot.segment<2>(xGOAL_POS) =
      R_v_g.transpose() * vel_g - I_2x3 * R_I_b.transpose() * vel_b;
  xdot(xGOAL_RHO) = rho_g * rho_g * e3.transpose() * R_I_b.transpose() * vel_b;
  // xdot.segment<2>(xGOAL_VEL) = 0.;
  xdot(xGOAL_ATT) = omega_g;
  // xdot(xGOAL_OMEGA) = 0.;

  // Landing goal jacobian
  // d goal / d UAV
  A.block<2, 1>(xGOAL_POS, xATT + 0) = -I_2x3 * d_R_d_phi.transpose() * vel_b;
  A.block<2, 1>(xGOAL_POS, xATT + 1) = -I_2x3 * d_R_d_theta.transpose() * vel_b;
  A.block<2, 1>(xGOAL_POS, xATT + 2) = -I_2x3 * d_R_d_psi.transpose() * vel_b;
  A.block<2, 3>(xGOAL_POS, xVEL) = -I_2x3 * R_I_b.transpose();

  A(xGOAL_RHO, xATT + 0) =
      rho_g * rho_g * e3.transpose() * d_R_d_phi.transpose() * vel_b;
  A(xGOAL_RHO, xATT + 1) =
      rho_g * rho_g * e3.transpose() * d_R_d_theta.transpose() * vel_b;
  A(xGOAL_RHO, xATT + 2) =
      rho_g * rho_g * e3.transpose() * d_R_d_psi.transpose() * vel_b;
  A.block<1, 3>(xGOAL_RHO, xVEL) =
      rho_g * rho_g * e3.transpose() * R_I_b.transpose();

  // d goal / d goal
  A.block<2, 2>(xGOAL_POS, xGOAL_VEL) = R_v_g.transpose();
  A.block<2, 1>(xGOAL_POS, xGOAL_ATT) = dR_v_g_dTheta.transpose() * vel_g;

  A(xGOAL_RHO, xGOAL_RHO) =
      2. * rho_g * e3.transpose() * R_I_b.transpose() * vel_b;

  A(xGOAL_ATT, xGOAL_OMEGA) = 1.;

  // Zero Landmark dynamics
  // for (unsigned int i = 0; i < MAXLANDMARKS; i++)
  //{
  // const unsigned int xRHOI = xGOAL_LM + 2 + 3 * i;
  // const double rho_i = x(xRHOI);

  //// dynamics
  // xdot(xRHOI) = rho_i * rho_i * e3.transpose() * R_I_b.transpose() * vel_b;

  //// jacobian
  //// d goal / d UAV
  // A(xRHOI, xATT + 0) = rho_i * rho_i * e3.transpose() * d_R_d_phi.transpose()
  // * vel_b; A(xRHOI, xATT + 1) = rho_i * rho_i * e3.transpose() *
  // d_R_d_theta.transpose() * vel_b; A(xRHOI, xATT + 2) = rho_i * rho_i *
  // e3.transpose() * d_R_d_psi.transpose() * vel_b; A.block<1, 3>(xRHOI, xVEL)
  // = rho_i * rho_i * e3.transpose() * R_I_b.transpose();

  //// d goal / d goal
  // A(xRHOI, xRHOI) = 2. * rho_i * e3.transpose() * R_I_b.transpose() * vel_b;
  //}
}

void Estimator::printLmIDs()
{
  std::list<int>::iterator it;

  std::cout << "List: ";
  for (it = landmark_ids_.begin(); it != landmark_ids_.end(); it++)
  {
    std::cout << *it << ", ";
  }
  std::cout << std::endl;
}

void Estimator::printLmXhat()
{
  std::cout << "lm xhat: " << std::endl
            << xhat_.bottomRows(MAXLANDMARKS * 3) << std::endl;
}

void Estimator::printLmPhat()
{
  std::cout << "lm Phat: " << std::endl
            << P_.bottomRightCorner(MAXLANDMARKS * 3, MAXLANDMARKS * 3)
            << std::endl;
}

void Estimator::initLandmark(const int& id, const Vector2d& pix)
{
  // lm_idx is 0 indexed
  const int lm_idx = landmark_ids_.size();

  // add the id to the list of ids tracked
  landmark_ids_.push_back(id);

  // initialize estimator xhat and phat
  const int xLM_IDX = xGOAL_LM + 3 * lm_idx;
  xhat_.block<3, 1>(xLM_IDX, 0) = x0_landmarks_;
  P_.block<3, 3>(xLM_IDX, xLM_IDX) = P0_landmarks_.asDiagonal();

  const Eigen::Matrix3d R_b_c = q_b_c_.R();

  const double phi = xhat_(xATT + 0);
  const double theta = xhat_(xATT + 1);
  const double psi = xhat_(xATT + 2);
  const Eigen::Matrix3d R_I_b = rotmItoB(phi, theta, psi);

  Eigen::Vector3d pix_homo(pix(0), pix(1), 1.);
  Eigen::Vector3d unit_vec_veh_frame =
      R_I_b.transpose() * R_b_c.transpose() * cam_K_inv_ * pix_homo;

  // TODO assumes zero body to camera postion offset
  const Eigen::Vector3d p_c_b_I = R_I_b * p_b_c_;
  //const double expected_altitude = 1. / xhat_(xGOAL_RHO);
  const double expected_altitude = 1. / xhat_(xGOAL_RHO) - p_c_b_I(2);

  Eigen::Vector3d scaled_vec_veh_frame =
      (expected_altitude / unit_vec_veh_frame(2)) * unit_vec_veh_frame;
  //Eigen::Vector3d p_i_v_v = scaled_vec_veh_frame;
  Eigen::Vector3d p_i_c_v = scaled_vec_veh_frame;
  const Eigen::Vector3d p_i_v_v = p_i_c_v + p_c_b_I;

  const double theta_g = xhat_(xGOAL_ATT);
  const Eigen::Matrix3d R_v_g = rotm3dItoB(theta_g);
  Eigen::Vector3d p_g_v_v;
  p_g_v_v(0) = xhat_(xGOAL_POS + 0);
  p_g_v_v(1) = xhat_(xGOAL_POS + 1);
  p_g_v_v(2) = 1. / xhat_(xGOAL_RHO);
  Eigen::Vector3d p_i_g_g = R_v_g * (p_i_v_v - p_g_v_v);

  // Init state with estimate
  xhat_.block<3, 1>(xLM_IDX, 0) = p_i_g_g;
}

void Estimator::removeLandmark(const int& lm_idx, const int& lm_id)
{
  landmark_ids_.remove(lm_id);

  // Move up the bottom rows to cover up the values corresponding to the removed
  // lm
  const int xLM_IDX = xGOAL_LM + 3 * lm_idx;
  const int num_rows = xZ - xLM_IDX - 3;
  const int num_cols = num_rows;

  xhat_.block(xLM_IDX, 0, num_rows, 1) = xhat_.bottomRows(num_rows);
  // Zero out the unintialized xhat terms (NOT necessary)
  xhat_.bottomRows(3).setZero();

  // Move covariance up and then left to preserve cross terms
  P_.block(xLM_IDX, 0, num_rows, xZ) = P_.bottomRightCorner(num_rows, xZ);
  P_.block(0, xLM_IDX, xZ, num_cols) = P_.bottomRightCorner(xZ, num_cols);
  // Zero out the unintialized covariance terms (necessary)
  P_.bottomRightCorner(3, xZ).setZero();
  P_.bottomRightCorner(xZ, 3).setZero();
}

void Estimator::updateLandmark(const int& idx, const Vector2d& pix,
                               const Matrix2d& R_pix)
{
  int lm_pix_dims = 0;
  MeasVec lm_pix_zhat;
  landmarkPixelMeasModel(idx, xhat_, lm_pix_dims, lm_pix_zhat, H_, cam_K_,
                         p_b_c_, q_b_c_);
  z_resid_.head(lm_pix_dims) = pix - lm_pix_zhat.head(lm_pix_dims);
  z_R_.topLeftCorner(lm_pix_dims, lm_pix_dims) = R_pix;
  update(lm_pix_dims, z_resid_, z_R_, H_);
}
