#include <iostream>

#include "landing_sim/utils.h"
#include "multirotor_sim/utils.h"

#include "estimator.h"
#include "estimator_utils.h"

using namespace Eigen;
using namespace multirotor_sim;

Estimator::Estimator(std::string filename)
{
  get_yaml_node("draw_feature_img", filename, draw_feats_);
  get_yaml_node("record_video", filename, record_vid_);
  get_yaml_node("use_goal_stop_time", filename, use_goal_stop_time_);

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
  Vector3d att_R_diag;
  get_yaml_eigen("att_R_diag", filename, att_R_diag);
  att_R_ = att_R_diag.asDiagonal();

  Vector2d pix_R_diag;
  get_yaml_eigen("pix_R_diag", filename, pix_R_diag);
  pix_R_ = pix_R_diag.asDiagonal();

  get_yaml_eigen("depth_R", filename, depth_R_);
  get_yaml_node("update_goal_depth", filename, update_goal_depth_);

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

  get_yaml_node("min_feat_depth", filename, min_depth_);
  setInverseDepth();

  last_time_ = -1;
  feats_initialized_ = false;

  q_I_b_ = quat::Quatd::Identity();
  u_in_.setZero();

  PRINTMAT(xhat_);
  PRINTMAT(P_);
  PRINTMAT(Q_);
}

Estimator::~Estimator()
{
}

void Estimator::imuCallback(const double& t, const Vector6d& z,
                            const Matrix6d& R)
{
  u_in_.block<3, 1>(uOMEGA, 0) = z.block<3, 1>(3, 0);
}

void Estimator::mocapCallback(const double& t, const Xformd& z,
                              const Matrix6d& R)
{
  q_I_b_ = z.q();
  q_I_b_.normalize();

  const Vector3d mocap_euler = z.q().euler();
  this->updateUAVAttitude(mocap_euler);
}

void Estimator::velocityCallback(const double& t, const Vector3d& vel_b,
                                 const Matrix3d& R)
{
  if (last_time_ < 0)
  {
    last_time_ = t;
    return;
  }

  Vector3d vel_I = q_I_b_.R().transpose() * vel_b;
  u_in_.block<3, 1>(uVEL, 0) = vel_I;

  const double dt = t - last_time_;
  propagate(dt, u_in_);

  last_time_ = t;
}

void Estimator::simpleCamCallback(const double& t, const ImageFeat& z,
                                  const Matrix2d& R_pix,
                                  const Matrix1d& R_depth)
{
  if (draw_feats_)
    drawImageFeatures(record_vid_, z.pixs);

  if (!feats_initialized_)
  {
    initLandmarks(z.pixs);
    feats_initialized_ = true;
    return;
  }

  if (t < use_goal_stop_time_)
  {
    //updateGoal(virtualImagePixels(z.pixs[0]));
    // update goal with real measured pixels, not virtual image
    updateGoal(z.pixs[0]);

    if (update_goal_depth_)
      updateGoalDepth(z.depths[0]);
  }

  // TODO put the landmark updates back
  //for (unsigned int i = 0; i < SIZE; i++)
  //{
    //unsigned int lm_id = i;
    //unsigned int lm_idx = i + 1;
    //// updateLandmark(lm_id, z.pixs[lm_idx]);
    //updateLandmark(lm_id, virtualImagePixels(z.pixs[lm_idx]));
  //}
}

void Estimator::initLandmarks(
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>& pixs)
{
  // Init Landmarks
  for (unsigned int i = 0; i < SIZE; i++)
  {
    unsigned int lm_id = i;
    unsigned int lm_idx = i + 1;

    double rho_init = 1. / (2. * min_depth_);
    Vector2d r_b = invserseMeasurementModelLandmark(pixs[lm_idx], rho_init);

    unsigned int rx_idx = xLM + 0 + 3 * i;
    xhat_.block<2, 1>(rx_idx, 0) = r_b;
  }

  setInverseDepth();
}

void Estimator::setInverseDepth()
{
  double rho_init = 1. / (2. * min_depth_);
  double cov_init = 1. / (16. * min_depth_ * min_depth_);

  xhat_(xRHO) = rho_init;
  P_(xRHO, xRHO) = cov_init;

  for (unsigned int i = 0; i < SIZE; i++)
  {
    unsigned int rho_idx = xLM + 2 + 3 * i;
    xhat_(rho_idx) = rho_init;
    P_(rho_idx, rho_idx) = cov_init;
  }
}

void Estimator::toRot(const double theta, Matrix2d& R)
{
  R << cos(theta), sin(theta), -sin(theta), cos(theta);
}

Matrix2d Estimator::dtheta_R(const double theta)
{
  Matrix2d dR;
  dR << -sin(theta), cos(theta), -cos(theta), -sin(theta);

  return dR;
}

void Estimator::propagate(const double& time_step, const InputVec& u_in)
{
  const double dt = time_step / num_prop_steps_;

  for (unsigned int i = 0; i < num_prop_steps_; i++)
  {
    xdot_.setZero();
    A_.setZero();

    const double uav_phi = xhat_(xUAVATT + 0);
    const double uav_theta = xhat_(xUAVATT + 1);
    const double uav_psi = xhat_(xUAVATT + 2);

    const double rho0 = xhat_(xRHO, 0);
    const Vector2d vel_b = xhat_.block<2, 1>(xVEL, 0);
    const double theta = xhat_(xATT);
    const double omega = xhat_(xOMEGA);

    const Vector2d vel_c_I_xy = u_in.block<2, 1>(uVEL, 0);
    const double vel_c_I_z = u_in(uVEL + 2);

    const double omega_p = u_in(uOMEGA + 0);
    const double omega_q = u_in(uOMEGA + 1);
    const double omega_r = u_in(uOMEGA + 2);

    Matrix2d R_I_b;
    toRot(theta, R_I_b);

    // UAV state dynamics and jacobians
    // phi, theta, psi
    const double sp = sin(uav_phi);
    const double cp = cos(uav_phi);
    const double ct = cos(uav_theta);
    const double tt = tan(uav_theta);

    xdot_(xUAVATT + 0) = omega_p + (sp * tt) * omega_q + (cp * tt) * omega_r;
    xdot_(xUAVATT + 1) = cp * omega_q - sp * omega_r;
    xdot_(xUAVATT + 2) = (sp / ct) * omega_q + (cp / ct) * omega_r;

    A_(xUAVATT + 0, xUAVATT + 0) = (cp * tt) * omega_q - (sp * tt) * omega_r;
    A_(xUAVATT + 0, xUAVATT + 1) =
        (sp / ct / ct) * omega_q + (cp / ct / ct) * omega_r;
    // A_(xUAVATT + 0, xUAVATT + 2) = 0;

    A_(xUAVATT + 1, xUAVATT + 0) = -sp * omega_q - cp * omega_r;
    // A_(xUAVATT + 1, xUAVATT + 1) = 0;
    // A_(xUAVATT + 1, xUAVATT + 2) = 0;

    A_(xUAVATT + 2, xUAVATT + 0) = (cp / ct) * omega_q - (sp / ct) * omega_r;
    A_(xUAVATT + 2, xUAVATT + 1) =
        (sp / ct) * tt * omega_q + (cp / ct) * tt * omega_r;
    // A_(xUAVATT + 2, xUAVATT + 2) = 0;

    // Vehicle state dynamics and jacobians
    // xdot_.block<2, 1>(xPOS, 0) = R_I_b.transpose() * vel_b; // - uav_vel(x,
    // y);
    xdot_.block<2, 1>(xPOS, 0) = R_I_b.transpose() * vel_b - vel_c_I_xy;
    xdot_(xRHO) = rho0 * rho0 * vel_c_I_z;
    xdot_(xATT) = omega;

    A_.block<2, 2>(xPOS, xVEL) = R_I_b.transpose();
    A_.block<2, 1>(xPOS, xATT) = dtheta_R(theta).transpose() * vel_b;
    A_(xRHO, xRHO) = 2. * rho0 * vel_c_I_z;
    A_(xATT, xOMEGA) = 1.;

    // Landmark state dynamics and jacobians
    for (unsigned int i = 0; i < SIZE; i++)
    {
      // Note, landmark offset vector r has 0 dynamics and 0 jacobians
      unsigned int rho_idx = xLM + 2 + 3 * i;
      double rhoi = xhat_(rho_idx);
      xdot_(rho_idx) = rhoi * rhoi * vel_c_I_z;
      A_(rho_idx, rho_idx) = 2. * rhoi * vel_c_I_z;
    }

    xhat_ += xdot_ * dt;

    A_ = I_ + A_ * dt;
    P_ = A_ * P_ * A_.transpose() + Q_;
  }
}

void Estimator::updateUAVAttitude(const Vector3d& uav_euler)
{
  att_residual_ = uav_euler - xhat_.block<3, 1>(xUAVATT, 0);

  att_H_.setZero();
  att_H_.block<3, 3>(0, xUAVATT).setIdentity();

  att_K_ = P_ * att_H_.transpose() *
           (att_H_ * P_ * att_H_.transpose() + att_R_).inverse();

  xhat_ += att_K_ * att_residual_;
  // just reuse A_ to save memory
  A_ = I_ - att_K_ * att_H_;
  P_ = A_ * P_ * A_.transpose() + att_K_ * att_R_ * att_K_.transpose();
}

void Estimator::updateGoal(const Vector2d& goal_pix_meas)
{
  const Vector2d goal_pix = virtualImagePixels(goal_pix_meas);

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

  Vector3d pos_I;
  pos_I(0) = xhat_(xPOS + 0);
  pos_I(1) = xhat_(xPOS + 1);
  pos_I(2) = 1. / rho0;

  Matrix3d Rot_I_b = R_I_b();

  Vector3d pos_cam = q_b_c_.R() * Rot_I_b * pos_I;

  //PRINTMAT(pos_I);
  //PRINTMAT(pos_cam);

  double u_cam = fx_ * pos_cam(0) / pos_cam(2) + cx_;
  double u_meas = goal_pix_meas(0);

  double v_cam = fy_ * pos_cam(1) / pos_cam(2) + cy_;
  double v_meas = goal_pix_meas(1);

  //std::cout << "u pix diff: " << u_cam - u_meas << std::endl;
  std::cout << "v pix diff: " << v_cam - v_meas << std::endl;

  //Matrix3d K_cam;
  //K_cam.setZero();
  //K_cam(0, 0) = fx_;
  //K_cam(1, 1) = fy_;
  //K_cam(2, 2) = 1.;
  //K_cam(0, 2) = cx_;
  //K_cam(1, 2) = cy_;

  //Vector3d est_pix_vec = K_cam * q_b_c_.R() * Rot_I_b * vec_I;
  //est_pix_vec /= est_pix_vec(2);
  //static const Vector3d e_13(1., 0., 0.);
  //static const Vector3d e_23(0., 1., 0.);
  //est_pix_vec(0) = fx_ * rho0 * e_13.transpose() * q_b_c_.R() * Rot_I_b * vec_I + cx_;
  //est_pix_vec(1) = fy_ * rho0 * e_23.transpose() * q_b_c_.R() * Rot_I_b * vec_I + cy_;

  //Vector2d diff_pix = goal_pix_meas - est_pix_vec.block<2, 1>(0, 0);
  //PRINTMAT(diff_pix);
  //PRINTMAT(goal_pix_meas);
  //PRINTMAT(est_pix_vec);
  //std::cout << "u pix diff: " << est_pix_vec(0) - goal_pix_meas(0) << std::endl;
  //std::cout << "v pix diff: " << est_pix_vec(1) - goal_pix_meas(1) << std::endl;

  pix_H_.setZero();
  pix_H_.block<1, 2>(0, xPOS) = fx_ * rho0 * e_1.transpose() * R_v_c;
  pix_H_(0, xRHO) = fx_ * e_1.transpose() * R_v_c * pos_v;
  pix_H_.block<1, 2>(1, xPOS) = fy_ * rho0 * e_2.transpose() * R_v_c;
  pix_H_(1, xRHO) = fy_ * e_2.transpose() * R_v_c * pos_v;

  pix_K_ = P_ * pix_H_.transpose() *
       (pix_H_ * P_ * pix_H_.transpose() + pix_R_).inverse();

  xhat_ += pix_K_ * pix_residual_;
  // just reuse A_ to save memory
  A_ = I_ - pix_K_ * pix_H_;
  P_ = A_ * P_ * A_.transpose() + pix_K_ * pix_R_ * pix_K_.transpose();
}

void Estimator::updateGoalDepth(const double& goal_depth)
{
  const double rho0 = xhat_(xRHO);
  Matrix1d depth_residual;
  depth_residual(0) = goal_depth - (1. / rho0);

  depth_H_.setZero();
  depth_H_(0, xRHO) = -1. / (rho0 * rho0);

  pix_K_.leftCols(1) = P_ * depth_H_.transpose() *
                   (depth_H_ * P_ * depth_H_.transpose() + depth_R_).inverse();

  xhat_ += pix_K_.leftCols(1) * depth_residual;
  // just reuse A_ to save memory
  A_ = I_ - pix_K_.leftCols(1) * depth_H_;
  P_ = A_ * P_ * A_.transpose() +
       pix_K_.leftCols(1) * depth_R_ * pix_K_.leftCols(1).transpose();
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
  pix_H_(0, xATT) =
      fx_ * rhoi * e_1.transpose() * R_v_c * dtheta_R(theta).transpose() * r_b;
  pix_H_(1, xATT) =
      fy_ * rhoi * e_2.transpose() * R_v_c * dtheta_R(theta).transpose() * r_b;

  // dpix / dr
  pix_H_.block<1, 2>(0, rx_idx) =
      fx_ * rhoi * e_1.transpose() * R_v_c * R_I_b.transpose();
  pix_H_.block<1, 2>(1, rx_idx) =
      fy_ * rhoi * e_2.transpose() * R_v_c * R_I_b.transpose();

  // dpix / drho
  pix_H_(0, rho_idx) = fx_ * e_1.transpose() * R_v_c * posi_v;
  pix_H_(1, rho_idx) = fy_ * e_2.transpose() * R_v_c * posi_v;

  // PRINTMAT(pix_H_);

  pix_K_ = P_ * pix_H_.transpose() *
       (pix_H_ * P_ * pix_H_.transpose() + pix_R_).inverse();

  xhat_ += pix_K_ * pix_residual_;
  // just reuse A_ to save memory
  A_ = I_ - pix_K_ * pix_H_;
  P_ = A_ * P_ * A_.transpose() + pix_K_ * pix_R_ * pix_K_.transpose();
}

void Estimator::updateLandmarkDepth(const int& id, const double& lm_depth)
{
  const unsigned int rho_idx = xLM + 2 + 3 * id;

  const double rhoi = xhat_(rho_idx);
  Matrix1d depth_residual;
  depth_residual(0) = lm_depth - (1. / rhoi);

  depth_H_.setZero();
  depth_H_(0, rho_idx) = -1. / (rhoi * rhoi);

  pix_K_.leftCols(1) = P_ * depth_H_.transpose() *
                   (depth_H_ * P_ * depth_H_.transpose() + depth_R_).inverse();

  xhat_ += pix_K_.leftCols(1) * depth_residual;
  // just reuse A_ to save memory
  A_ = I_ - pix_K_.leftCols(1) * depth_H_;
  P_ = A_ * P_ * A_.transpose() +
       pix_K_.leftCols(1) * depth_R_ * pix_K_.leftCols(1).transpose();
}

Vector2d Estimator::invserseMeasurementModelLandmark(const Vector2d& lm_pix,
                                                     const double rho)
{
  static const Matrix2d R_v_c = q_b_c_.R().block<2, 2>(0, 0);

  const Vector2d pos_v = xhat_.block<2, 1>(xPOS, 0);
  const double theta = xhat_(xATT);

  Matrix2d R_I_b;
  toRot(theta, R_I_b);

  Vector2d posi_c;
  posi_c(0) = (1. / fx_) * (lm_pix(0) - cx_);
  posi_c(1) = (1. / fy_) * (lm_pix(1) - cy_);

  Vector2d r_b = R_I_b * ((1. / rho) * R_v_c.transpose() * posi_c - pos_v);

  return r_b;
}

Vector2d Estimator::virtualImagePixels(const Vector2d& pix)
{
  Vector3d pix_vec(pix(0), pix(1), 1.);

  Matrix3d K;
  K << fx_, 0., cx_, 0., fy_, cy_, 0., 0., 1.;

  pix_vec = K.inverse() * pix_vec;

  static const Matrix3d R_b_c = q_b_c_.R();

  Vector3d virtual_vec;
  virtual_vec =
      K * R_b_c * q_I_b_.R().transpose() * R_b_c.transpose() * pix_vec;

  Vector2d virtual_pix;
  virtual_pix = virtual_vec.block<2, 1>(0, 0) / virtual_vec(2);
  return virtual_pix;
}

Matrix3d Estimator::R_I_b() const
{
  const double phi = xhat_(xUAVATT + 0);
  const double theta = xhat_(xUAVATT + 1);
  const double psi = xhat_(xUAVATT + 2);

  return rotmItoB(phi, theta, psi);
}
