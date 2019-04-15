#include <iostream>

#include "landing_sim/utils.h"
#include "multirotor_sim/utils.h"

#include "estimator.h"

using namespace Eigen;
using namespace multirotor_sim;

Estimator::Estimator(std::string filename)
{
  get_yaml_node("draw_feature_img", filename, draw_feats_);

  // Filter init
  get_yaml_eigen("x0", filename, xhat_);

  StateVec P_diag;
  get_yaml_eigen("P0_diag", filename, P_diag);
  P_ = P_diag.asDiagonal();

  // Prop vars
  get_yaml_node("num_prop_steps", filename, num_prop_steps_);
  StateVec Q_diag;
  get_yaml_eigen("Q_diag", filename, Q_diag);
  Q_ = (1. / num_prop_steps_) * Q_diag.asDiagonal();

  // Update vars
  Vector2d pix_R_diag;
  get_yaml_eigen("pix_R_diag", filename, pix_R_diag);
  pix_R_ = pix_R_diag.asDiagonal();

  get_yaml_eigen("depth_R", filename, depth_R_);

  Vector2d focal_len;
  get_yaml_eigen("focal_len", filename, focal_len);
  fx_ = focal_len(0);
  fy_ = focal_len(1);

  Vector2d cam_center;
  get_yaml_eigen("cam_center", filename, cam_center);
  cx_ = cam_center(0);
  cy_ = cam_center(1);

  Vector4d q_b_c_eig;
  get_yaml_eigen("q_b_c", filename, q_b_c_eig);
  q_b_c_ = quat::Quatd(q_b_c_eig);

  last_time_ = -1;
  PRINTMAT(xhat_);
  PRINTMAT(P_);
  PRINTMAT(Q_);
}

Estimator::~Estimator()
{
}

void Estimator::mocapCallback(const double& t, const Xformd& z, const Matrix6d& R)
{
}

void Estimator::simpleCamCallback(const double& t, const ImageFeat& z,
                       const Matrix2d& R_pix, const Matrix1d& R_depth)
{
  if (draw_feats_)
    drawImageFeatures(z.pixs);

  if (last_time_ < 0)
  {
    last_time_ = t;
    return;
  }

  const double dt = t - last_time_;
  propagate(dt);

  updateGoal(z.pixs[0]);
  updateGoalDepth(z.depths[0]);

  for (unsigned int i = 0; i < SIZE; i++)
  {
    unsigned int lm_id = i;
    unsigned int lm_idx = i + 1;
    updateLandmark(lm_id, z.pixs[lm_idx]);
  }

  last_time_ = t;
}

void Estimator::toRot(const double theta, Matrix2d& R)
{
  R << cos(theta), sin(theta),
       -sin(theta), cos(theta);
}

Matrix2d Estimator::dtheta_R(const double theta)
{
  Matrix2d dR;
  dR << -sin(theta), cos(theta),
       -cos(theta), -sin(theta);

  return dR;
}

void Estimator::propagate(const double& time_step)
{
  const double dt = time_step / num_prop_steps_;
  
  for (unsigned int i = 0; i < num_prop_steps_; i++)
  {
    xdot_.setZero();
    A_.setZero();

    const double rho0 = xhat_(xRHO, 0);
    const Vector2d vel_b = xhat_.block<2, 1>(xVEL, 0);
    const double theta = xhat_(xATT);
    const double omega = xhat_(xOMEGA);

    Matrix2d R_I_b;
    toRot(theta, R_I_b);

    // Vehicle state dynamics and jacobians
    xdot_.block<2, 1>(xPOS, 0) = R_I_b.transpose() * vel_b; // - uav_vel(x, y);
    //xdot_(xRHO) = rho0 * rho0 * uav_vel_z;
    xdot_(xATT) = omega;

    A_.block<2, 2>(xPOS, xVEL) = R_I_b.transpose();
    A_.block<2, 1>(xPOS, xATT) = dtheta_R(theta).transpose() * vel_b;
    //A_(xRHO, xRHO) = 2. * rho0 * uav_vel_z;
    A_(xATT, xOMEGA) = 1.;

    // Landmark state dynamics and jacobians
    for (unsigned int i = 0; i < SIZE; i++)
    {
      // Note, landmark offset vector r has 0 dynamics and 0 jacobians
      unsigned int rho_idx = xLM + 2 + 3 * i;
      double rhoi = xhat_(rho_idx);
      //xdot_(rho_idx) = rhoi * rhoi * uav_vel_z;
      //A_(rho_idx, rho_idx) = 2. * rhoi * uav_vel_z;
    }

    xhat_ += xdot_ * dt;

    A_ = I_ + A_ * dt;
    P_ = A_ * P_ * A_.transpose() + Q_;
  }
}

void Estimator::updateGoal(const Vector2d& goal_pix)
{
  static const Vector2d e_1(1., 0.);
  static const Vector2d e_2(0., 1.);

  static const Matrix2d R_v_c = q_b_c_.R().block<2, 2>(0, 0);

  const Vector2d pos_v = xhat_.block<2, 1>(xPOS, 0);
  const double rho0 = xhat_(xRHO);
  const Vector2d pos_c = R_v_c * pos_v;

  Vector2d zhat;
  zhat(0) = fx_ * rho0 * pos_c(0) + cx_;
  zhat(1) = fy_ * rho0 * pos_c(1) + cy_;

  pix_residual_ = goal_pix - zhat;

  pix_H_.setZero();
  pix_H_.block<1, 2>(0, xPOS) = fx_ * rho0 * e_1.transpose() * R_v_c;
  pix_H_(0, xRHO) = fx_ * e_1.transpose() * R_v_c * pos_v;
  pix_H_.block<1, 2>(1, xPOS) = fy_ * rho0 * e_2.transpose() * R_v_c;
  pix_H_(1, xRHO) = fy_ * e_2.transpose() * R_v_c * pos_v;

  K_ = P_ * pix_H_.transpose() * (pix_H_ * P_ * pix_H_.transpose() + pix_R_).inverse();

  xhat_ += K_ * pix_residual_;
  // just reuse A_ to save memory
  A_ = I_ - K_ * pix_H_;
  P_ = A_ * P_ * A_.transpose() + K_ * pix_R_ * K_.transpose();
}

void Estimator::updateGoalDepth(const double& goal_depth)
{
  const double rho0 = xhat_(xRHO);
  Matrix1d depth_residual;
  depth_residual(0) = goal_depth - (1. / rho0);

  depth_H_.setZero();
  depth_H_(0, xRHO) = -1. / (rho0 * rho0);

  K_.leftCols(1) = P_ * depth_H_.transpose() *
                   (depth_H_ * P_ * depth_H_.transpose() + depth_R_).inverse();

  xhat_ += K_.leftCols(1) * depth_residual;
  // just reuse A_ to save memory
  A_ = I_ - K_.leftCols(1) * depth_H_;
  P_ = A_ * P_ * A_.transpose() +
       K_.leftCols(1) * depth_R_ * K_.leftCols(1).transpose();
}

void Estimator::updateLandmark(const int& id, const Vector2d& lm_pix)
{
  static const Vector2d e_1(1., 0.);
  static const Vector2d e_2(0., 1.);

  static const Matrix2d R_v_c = q_b_c_.R().block<2, 2>(0, 0);

  // Landmark ids are from 0 -> (SIZE - 1)
  const unsigned int rx_idx = xLM + 0 + 3 * id;
  const unsigned int ry_idx = xLM + 1 + 3 * id;
  const unsigned int rho_idx = xLM + 2 + 3 * id;

  const Vector2d pos_v = xhat_.block<2, 1>(xPOS, 0);
  const double theta = xhat_(xATT);
  const Vector2d r_b = xhat_.block<2, 1>(rx_idx, 0);
  const double rhoi = xhat_(rho_idx);

  Matrix2d R_I_b;
  toRot(theta, R_I_b);

  const Vector2d posi_v = pos_v + R_I_b.transpose() * r_b;
  const Vector2d posi_c = R_v_c * posi_v;

  Vector2d zhat;
  zhat(0) = fx_ * rhoi * posi_c(0) + cx_;
  zhat(1) = fy_ * rhoi * posi_c(1) + cy_;

  pix_residual_ = lm_pix - zhat;

  pix_H_.setZero();
  // dpix / dpos
  pix_H_.block<1, 2>(0, xPOS) = fx_ * rhoi * e_1.transpose() * R_v_c;
  pix_H_.block<1, 2>(1, xPOS) = fy_ * rhoi * e_2.transpose() * R_v_c;

  // dpix / dtheta
  pix_H_(0, xATT) = fx_ * rhoi * e_1.transpose() * R_v_c * dtheta_R(theta).transpose() * r_b;
  pix_H_(1, xATT) = fy_ * rhoi * e_2.transpose() * R_v_c * dtheta_R(theta).transpose() * r_b;

  // dpix / dr
  pix_H_.block<1, 2>(0, rx_idx) = fx_ * rhoi * e_1.transpose() * R_v_c * R_I_b.transpose();
  pix_H_.block<1, 2>(1, rx_idx) = fx_ * rhoi * e_2.transpose() * R_v_c * R_I_b.transpose();

  // dpix / drho
  pix_H_(0, rho_idx) = fx_ * e_1.transpose() * R_v_c * posi_v;
  pix_H_(1, rho_idx) = fy_ * e_2.transpose() * R_v_c * posi_v;

  K_ = P_ * pix_H_.transpose() * (pix_H_ * P_ * pix_H_.transpose() + pix_R_).inverse();

  xhat_ += K_ * pix_residual_;
  // just reuse A_ to save memory
  A_ = I_ - K_ * pix_H_;
  P_ = A_ * P_ * A_.transpose() + K_ * pix_R_ * K_.transpose();
}
