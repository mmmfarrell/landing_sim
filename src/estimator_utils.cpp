#include "estimator_utils.h"

#include <Eigen/Dense>

Eigen::Matrix3d rotmItoB(const double phi, const double theta, const double psi)
{
  const double cp = cos(phi);
  const double sp = sin(phi);
  const double ct = cos(theta);
  const double st = sin(theta);
  const double cpsi = cos(psi);
  const double spsi = sin(psi);

  // Inertial frame to body frame from UAV book
  Eigen::Matrix3d rotm;
  rotm(0, 0) = ct * cpsi;
  rotm(0, 1) = ct * spsi;
  rotm(0, 2) = -st;

  rotm(1, 0) = sp * st * cpsi - cp * spsi;
  rotm(1, 1) = sp * st * spsi + cp * cpsi;
  rotm(1, 2) = sp * ct;

  rotm(2, 0) = cp * st * cpsi + sp * spsi;
  rotm(2, 1) = cp * st * spsi - sp * cpsi;
  rotm(2, 2) = cp * ct;

  return rotm;
}

Eigen::Matrix3d dRIBdPhi(const double phi, const double theta, const double psi)
{
  const double cp = cos(phi);
  const double sp = sin(phi);
  const double ct = cos(theta);
  const double st = sin(theta);
  const double cpsi = cos(psi);
  const double spsi = sin(psi);

  // Inertial frame to body frame from UAV book
  Eigen::Matrix3d rotm;
  rotm(0, 0) = 0.;
  rotm(0, 1) = 0.;
  rotm(0, 2) = 0.;

  rotm(1, 0) = cp * st * cpsi + sp * spsi;
  rotm(1, 1) = cp * st * spsi - sp * cpsi;
  rotm(1, 2) = cp * ct;

  rotm(2, 0) = -sp * st * cpsi + cp * spsi;
  rotm(2, 1) = -sp * st * spsi - cp * cpsi;
  rotm(2, 2) = -sp * ct;

  return rotm;
}

Eigen::Matrix3d dRIBdTheta(const double phi, const double theta, const double psi)
{
  const double cp = cos(phi);
  const double sp = sin(phi);
  const double ct = cos(theta);
  const double st = sin(theta);
  const double cpsi = cos(psi);
  const double spsi = sin(psi);

  // Inertial frame to body frame from UAV book
  Eigen::Matrix3d rotm;
  rotm(0, 0) = -st * cpsi;
  rotm(0, 1) = -st * spsi;
  rotm(0, 2) = -ct;

  rotm(1, 0) = sp * ct * cpsi;
  rotm(1, 1) = sp * ct * spsi;
  rotm(1, 2) = -sp * st;

  rotm(2, 0) = cp * ct * cpsi;
  rotm(2, 1) = cp * ct * spsi;
  rotm(2, 2) = -cp * st;

  return rotm;
}

Eigen::Matrix3d dRIBdPsi(const double phi, const double theta, const double psi)
{
  const double cp = cos(phi);
  const double sp = sin(phi);
  const double ct = cos(theta);
  const double st = sin(theta);
  const double cpsi = cos(psi);
  const double spsi = sin(psi);

  // Inertial frame to body frame from UAV book
  Eigen::Matrix3d rotm;
  rotm(0, 0) = -ct * spsi;
  rotm(0, 1) = ct * cpsi;
  rotm(0, 2) = 0.;

  rotm(1, 0) = -sp * st * spsi - cp * cpsi;
  rotm(1, 1) = sp * st * cpsi - cp * spsi;
  rotm(1, 2) = 0.;

  rotm(2, 0) = -cp * st * spsi + sp * cpsi;
  rotm(2, 1) = cp * st * cpsi + sp * spsi;
  rotm(2, 2) = 0.;

  return rotm;
}

Eigen::Matrix2d rotm2dItoB(const double theta)
{
  const double ct = cos(theta);
  const double st = sin(theta);

  // Inertial frame to body frame from UAV book
  Eigen::Matrix2d rotm;
  rotm(0, 0) = ct;
  rotm(0, 1) = st;

  rotm(1, 0) = -st;
  rotm(1, 1) = ct;

  return rotm;
}

Eigen::Matrix2d dR2DdTheta(const double theta)
{
  const double ct = cos(theta);
  const double st = sin(theta);

  // Inertial frame to body frame from UAV book
  Eigen::Matrix2d rotm;
  rotm(0, 0) = -st;
  rotm(0, 1) = ct;

  rotm(1, 0) = -ct;
  rotm(1, 1) = -st;

  return rotm;
}

Eigen::Matrix3d rotm3dItoB(const double theta)
{
  const double ct = cos(theta);
  const double st = sin(theta);

  // Inertial frame to body frame from UAV book
  Eigen::Matrix3d rotm;
  rotm.setZero();

  rotm(0, 0) = ct;
  rotm(0, 1) = st;

  rotm(1, 0) = -st;
  rotm(1, 1) = ct;

  rotm(2, 2) = 1.;

  return rotm;
}

Eigen::Matrix3d dR3DdTheta(const double theta)
{
  const double ct = cos(theta);
  const double st = sin(theta);

  // Inertial frame to body frame from UAV book
  Eigen::Matrix3d rotm;
  rotm.setZero();

  rotm(0, 0) = -st;
  rotm(0, 1) = ct;

  rotm(1, 0) = -ct;
  rotm(1, 1) = -st;

  rotm(2, 2) = 0.;

  return rotm;
}

Eigen::Matrix3d wMat(const double phi, const double theta, const double psi)
{
  const double cp = cos(phi);
  const double sp = sin(phi);
  const double ct = cos(theta);
  const double tt = tan(theta);

  Eigen::Matrix3d wmat;
  wmat.setZero();

  wmat(0, 0) = 1.;
  wmat(0, 1) = sp * tt;
  wmat(0, 2) = cp * tt;

  wmat(1, 0) = 0.;
  wmat(1, 1) = cp;
  wmat(1, 2) = -sp;

  wmat(2, 0) = 0.;
  wmat(2, 1) = sp / ct;
  wmat(2, 2) = cp / ct;

  return wmat;
}

//Eigen::Vector2d goalPixelMeasModel(double phi, double theta, double psi,
                                   //double pos_x, double pos_y, double rho)
//{
  //Vector3d pos_I;
  //pos_I(0) = pos_x;
  //pos_I(1) = pos_y;
  //pos_I(2) = 1. / rho;

  //Matrix3d Rot_I_b = rotmItoB(phi, theta, psi);

  //Vector3d pos_cam = q_b_c_.R() * Rot_I_b * pos_I;

  //double u_cam = fx_ * pos_cam(0) / pos_cam(2) + cx_;
  //double u_meas = goal_pix_meas(0);

  //double v_cam = fy_ * pos_cam(1) / pos_cam(2) + cy_;
  //double v_meas = goal_pix_meas(1);
//}

