#include <iostream>

#include "landing_sim/utils.h"
#include "multirotor_sim/utils.h"

#include "uav_estimator.h"
#include "estimator_utils.h"

using namespace Eigen;
using namespace multirotor_sim;

Estimator::Estimator(std::string filename)
{
  // Init EKF State
  xhat_.setZero();
  P_.setIdentity();

  Q_.setIdentity();
  Q_ *= 0.001;

  last_prop_time_ = -999.;
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
}

void Estimator::mocapCallback(const double& t, const Xformd& z,
                              const Matrix6d& R)
{

  // Update atttitude
  const double att_dims = 3;
  const Vector3d mocap_euler = z.q().euler();
  z_resid_.head(att_dims) = mocap_euler - xhat_.segment<3>(xATT);
  z_R_.topLeftCorner(att_dims, att_dims) = 0.001 * Eigen::Matrix3d::Identity();

  H_.setZero();
  H_.block<3, 3>(0, xATT);

  update(att_dims, z_resid_, z_R_, H_);
  // this->updateUAVAttitude(mocap_euler);
}

void Estimator::velocityCallback(const double& t, const Vector3d& vel_b,
                                 const Matrix3d& R)
{
  // if (last_time_ < 0)
  //{
  // last_time_ = t;
  // return;
  //}

  // Vector3d vel_I = q_I_b_.R().transpose() * vel_b;
  // u_in_.block<3, 1>(uVEL, 0) = vel_I;

  // const double dt = t - last_time_;
  // propagate(dt, u_in_);

  // last_time_ = t;
}

void Estimator::simpleCamCallback(const double& t, const ImageFeat& z,
                                  const Matrix2d& R_pix,
                                  const Matrix1d& R_depth)
{
  // if (draw_feats_)
  // drawImageFeatures(record_vid_, z.pixs);

  // if (!feats_initialized_)
  //{
  // initLandmarks(z.pixs);
  // feats_initialized_ = true;
  // return;
  //}

  // if (t < use_goal_stop_time_)
  //{
  ////updateGoal(virtualImagePixels(z.pixs[0]));
  //// update goal with real measured pixels, not virtual image
  // updateGoal(z.pixs[0]);

  // if (update_goal_depth_)
  // updateGoalDepth(z.depths[0]);
  //}

  //// TODO put the landmark updates back
  ////for (unsigned int i = 0; i < SIZE; i++)
  ////{
  ////unsigned int lm_id = i;
  ////unsigned int lm_idx = i + 1;
  ////// updateLandmark(lm_id, z.pixs[lm_idx]);
  ////updateLandmark(lm_id, virtualImagePixels(z.pixs[lm_idx]));
  ////}
}

void Estimator::propagate(const double& dt, const InputVec& u_in)
{
  dynamics(xhat_, u_, xdot_, A_);

  xhat_ += xdot_ * dt;

  A_ = I_ + A_ * dt;
  P_ = A_ * P_ * A_.transpose() + Q_;
}

void Estimator::update(const double dims, const MeasVec& residual,
                       const MeasMat& R, const MeasH& H)
{
  //att_residual_ = uav_euler - xhat_.block<3, 1>(xUAVATT, 0);

  //att_H_.setZero();
  //att_H_.block<3, 3>(0, xUAVATT).setIdentity();

  //att_K_ = P_ * att_H_.transpose() *
           //(att_H_ * P_ * att_H_.transpose() + att_R_).inverse();

  K_.leftCols(dims) = P_ * H.topRows(dims).transpose() *
                      (H.topRows(dims) * P_ * H.topRows(dims).transpose() +
                       R.topLeftCorner(dims, dims))
                          .inverse();

  xhat_ += K_.leftCols(dims) * residual.head(dims);
  // just reuse A_ to save memory
  A_ = I_ - K_.leftCols(dims) * H.topRows(dims);
  P_ = A_ * P_ * A_.transpose() + K_.leftCols(dims) *
                                      R.topLeftCorner(dims, dims) *
                                      K_.leftCols(dims).transpose();
  //xhat_ += att_K_ * att_residual_;
  //A_ = I_ - att_K_ * att_H_;
  //P_ = A_ * P_ * A_.transpose() + att_K_ * att_R_ * att_K_.transpose();
}

void Estimator::dynamics(const StateVec& x, const InputVec& u_in,
                         StateVec& xdot, StateMat& A)
{
  const double phi = x(xATT + 0);
  const double theta = x(xATT + 1);
  const double psi = x(xATT + 2);
  const double u = x(xVEL + 0);
  const double v = x(xVEL + 1);
  const double w = x(xVEL + 2);
  const double mu = x(xMU);

  const double az = u_in(uAZ);
  const double p = u_in(uOMEGA + 0);
  const double q = u_in(uOMEGA + 1);
  const double r = u_in(uOMEGA + 2);

  const Eigen::Vector3d vel_b = x.segment<3>(xVEL);
  const Eigen::Vector3d pqr = u_in.segment<3>(uOMEGA);

  const Eigen::Matrix3d R_I_b = rotmItoB(phi, theta, psi);
  const Eigen::Matrix3d wmat = wMat(phi, theta, psi);

  static const Eigen::Vector3d grav_I(0., 0., GRAVITY);
  const Eigen::Vector3d grav_b = R_I_b * grav_I;

  // Dynamics
  xdot.setZero();
  xdot.segment<3>(xPOS) = R_I_b.transpose() * vel_b;
  xdot.segment<3>(xATT) = wmat * pqr;
  xdot(xVEL + 0) = grav_b(0) + v * r - w * q - mu * u;
  xdot(xVEL + 1) = grav_b(1) + w * p - u * r - mu * v;
  xdot(xVEL + 2) = grav_b(2) + u * q - v * p - az;
  xdot(xMU) = 0.;

  // Jacobian
  const double cp = cos(phi);
  const double sp = sin(phi);
  const double ct = cos(theta);
  const double st = sin(theta);
  const double tt = tan(theta);
  const double cpsi = cos(psi);
  const double spsi = sin(psi);

  // Pos dot
  // d pdot / d att
  A(xPOS + 0, xATT + 0) =
      (cp * st * cpsi + sp * spsi) * v + (-sp * st * cpsi + cp * spsi) * w;
  A(xPOS + 0, xATT + 1) =
      (-st * cpsi) * u + (sp * ct * cpsi) * v + (cp * ct * cpsi) * w;
  A(xPOS + 0, xATT + 2) = (-ct * spsi) * u + (-sp * st * spsi - cp * cpsi) * v +
                          (-cp * st * spsi + sp * cpsi) * w;

  A(xPOS + 1, xATT + 0) =
      (cp * st * spsi - sp * cpsi) * v + (-sp * st * spsi - cp * cpsi) * w;
  A(xPOS + 1, xATT + 1) =
      (-st * spsi) * u + (sp * ct * spsi) * v + (cp * ct * spsi) * w;
  A(xPOS + 1, xATT + 2) = (ct * cpsi) * u + (sp * st * cpsi - cp * spsi) * v +
                          (cp * st * cpsi + sp * spsi) * w;

  A(xPOS + 2, xATT + 0) = (cp * ct) * v + (-sp * ct) * w;
  A(xPOS + 2, xATT + 1) = (-ct) * u + (-sp * st) * v + (-cp * st) * w;
  A(xPOS + 2, xATT + 2) = 0.;

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
}

