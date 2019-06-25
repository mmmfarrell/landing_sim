#include "measurement_models.h"

#include "landing_estimator.h"
#include "estimator_utils.h"

void goalDepthMeasModel(const Estimator::StateVec& x, int& meas_dims,
                        Estimator::MeasVec& z, Estimator::MeasH& H)
{
  meas_dims = 1;
  z.setZero();
  H.setZero();

  // Camera Params
  const double fx = 410.;
  const double fy = 420.;
  const double cx = 320.;
  const double cy = 240.;

  // Constants
  static const Eigen::Vector3d e1(1., 0., 0.);
  static const Eigen::Vector3d e2(0., 1., 0.);
  static const Eigen::Vector3d e3(0., 0., 1.);
  Eigen::Vector4d q(0.7071, 0., 0., 0.7071);
  q /= q.norm();
  quat::Quatd q_b_c(q);
  const Eigen::Matrix3d R_b_c = q_b_c.R();

  const double phi = x(Estimator::xATT + 0);
  const double theta = x(Estimator::xATT + 1);
  const double psi = x(Estimator::xATT + 2);
  const Eigen::Matrix3d R_I_b = rotmItoB(phi, theta, psi);

  const double rho_g = x(Estimator::xGOAL_RHO);
  Eigen::Vector3d p_g_v_v;
  p_g_v_v(0) = x(Estimator::xGOAL_POS + 0);
  p_g_v_v(1) = x(Estimator::xGOAL_POS + 1);
  p_g_v_v(2) = 1. / rho_g;

  const Eigen::Vector3d p_g_c_c = R_b_c * R_I_b * p_g_v_v;

  // Measurement Model
  z(0) = p_g_c_c(2);

  // Measurement Model Jacobian
  const Eigen::Matrix3d d_R_d_phi = dRIBdPhi(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_theta = dRIBdTheta(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_psi = dRIBdPsi(phi, theta, psi);

  H.setZero();
  H(0, Estimator::xATT + 0) = e3.transpose() * R_b_c * d_R_d_phi * p_g_v_v;
  H(0, Estimator::xATT + 1) = e3.transpose() * R_b_c * d_R_d_theta * p_g_v_v;
  H(0, Estimator::xATT + 2) = e3.transpose() * R_b_c * d_R_d_psi * p_g_v_v;

  const Eigen::Vector3d dzdp = e3.transpose() * R_b_c * R_I_b;
  H.block<1, 2>(0, Estimator::xGOAL_POS) = dzdp.head(2);
  H(0, Estimator::xGOAL_RHO) = (-1. / rho_g / rho_g) * dzdp(2);
}

void goalPixelMeasModel(const Estimator::StateVec& x, int& meas_dims,
                        Estimator::MeasVec& z, Estimator::MeasH& H)
{
  meas_dims = 2;
  z.setZero();
  H.setZero();

  // Camera Params
  const double fx = 410.;
  const double fy = 420.;
  const double cx = 320.;
  const double cy = 240.;

  // Constants
  static const Eigen::Vector3d e1(1., 0., 0.);
  static const Eigen::Vector3d e2(0., 1., 0.);
  static const Eigen::Vector3d e3(0., 0., 1.);
  Eigen::Vector4d q(0.7071, 0., 0., 0.7071);
  q /= q.norm();
  quat::Quatd q_b_c(q);
  const Eigen::Matrix3d R_b_c = q_b_c.R();

  const double phi = x(Estimator::xATT + 0);
  const double theta = x(Estimator::xATT + 1);
  const double psi = x(Estimator::xATT + 2);
  const Eigen::Matrix3d R_I_b = rotmItoB(phi, theta, psi);

  const double rho_g = x(Estimator::xGOAL_RHO);
  Eigen::Vector3d p_g_v_v;
  p_g_v_v(0) = x(Estimator::xGOAL_POS + 0);
  p_g_v_v(1) = x(Estimator::xGOAL_POS + 1);
  p_g_v_v(2) = 1. / rho_g;

  const Eigen::Vector3d p_g_c_c = R_b_c * R_I_b * p_g_v_v;

  // Measurement Model
  const double px_hat = fx * (p_g_c_c(0) / p_g_c_c(2)) + cx;
  const double py_hat = fy * (p_g_c_c(1) / p_g_c_c(2)) + cy;
  z(0) = px_hat;
  z(1) = py_hat;
  // const Eigen::Vector2d zhat(px_hat, py_hat);

  // Measurement Model Jacobian
  const Eigen::Matrix3d d_R_d_phi = dRIBdPhi(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_theta = dRIBdTheta(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_psi = dRIBdPsi(phi, theta, psi);

  const Eigen::Vector3d RdRdPhip = R_b_c * d_R_d_phi * p_g_v_v;
  const double dpx_dphi =
      (fx * RdRdPhip(0) / p_g_c_c(2)) -
      (fx * RdRdPhip(2) * p_g_c_c(0) / p_g_c_c(2) / p_g_c_c(2));
  const Eigen::Vector3d RdRdThetap = R_b_c * d_R_d_theta * p_g_v_v;
  const double dpx_dtheta =
      (fx * RdRdThetap(0) / p_g_c_c(2)) -
      (fx * RdRdThetap(2) * p_g_c_c(0) / p_g_c_c(2) / p_g_c_c(2));
  const Eigen::Vector3d RdRdPsip = R_b_c * d_R_d_psi * p_g_v_v;
  const double dpx_dpsi =
      (fx * RdRdPsip(0) / p_g_c_c(2)) -
      (fx * RdRdPsip(2) * p_g_c_c(0) / p_g_c_c(2) / p_g_c_c(2));

  const Eigen::Vector3d dpx_dp =
      ((fx * e1.transpose() * R_b_c * R_I_b) / p_g_c_c(2)) -
      ((fx * e3.transpose() * R_b_c * R_I_b * p_g_c_c(0)) /
       (p_g_c_c(2) * p_g_c_c(2)));
  const double dpx_drho = -(1. / rho_g / rho_g) * dpx_dp(2);

  H.setZero();
  H(0, Estimator::xATT + 0) = dpx_dphi;
  H(0, Estimator::xATT + 1) = dpx_dtheta;
  H(0, Estimator::xATT + 2) = dpx_dpsi;
  H.block<1, 2>(0, Estimator::xGOAL_POS) = dpx_dp.head(2);
  H(0, Estimator::xGOAL_RHO) = dpx_drho;

  const double dpy_dphi =
      (fy * RdRdPhip(1) / p_g_c_c(2)) -
      (fy * RdRdPhip(2) * p_g_c_c(1) / p_g_c_c(2) / p_g_c_c(2));
  const double dpy_dtheta =
      (fy * RdRdThetap(1) / p_g_c_c(2)) -
      (fy * RdRdThetap(2) * p_g_c_c(1) / p_g_c_c(2) / p_g_c_c(2));
  const double dpy_dpsi =
      (fy * RdRdPsip(1) / p_g_c_c(2)) -
      (fy * RdRdPsip(2) * p_g_c_c(1) / p_g_c_c(2) / p_g_c_c(2));

  const Eigen::Vector3d dpy_dp =
      ((fy * e2.transpose() * R_b_c * R_I_b) / p_g_c_c(2)) -
      ((fy * e3.transpose() * R_b_c * R_I_b * p_g_c_c(1)) /
       (p_g_c_c(2) * p_g_c_c(2)));
  const double dpy_drho = -(1. / rho_g / rho_g) * dpy_dp(2);

  H(1, Estimator::xATT + 0) = dpy_dphi;
  H(1, Estimator::xATT + 1) = dpy_dtheta;
  H(1, Estimator::xATT + 2) = dpy_dpsi;
  H.block<1, 2>(1, Estimator::xGOAL_POS) = dpy_dp.head(2);
  H(1, Estimator::xGOAL_RHO) = dpy_drho;
}
