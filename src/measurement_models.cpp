#include "measurement_models.h"

#include "landing_estimator.h"
#include "estimator_utils.h"

void goalDepthMeasModel(const Estimator::StateVec& x, int& meas_dims,
                        Estimator::MeasVec& z, Estimator::MeasH& H,
                        const Eigen::Vector3d& p_b_c, const quat::Quatd& q_b_c)
{
  meas_dims = 1;
  z.setZero();
  H.setZero();

  // Camera Params
  //const double fx = 410.;
  //const double fy = 420.;
  //const double cx = 320.;
  //const double cy = 240.;

  // Constants
  static const Eigen::Vector3d e1(1., 0., 0.);
  static const Eigen::Vector3d e2(0., 1., 0.);
  static const Eigen::Vector3d e3(0., 0., 1.);
  //Eigen::Vector4d q(0.7071, 0., 0., 0.7071);
  //q /= q.norm();
  //quat::Quatd q_b_c(q);
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

  const Eigen::Vector3d p_g_c_c = R_b_c * (R_I_b * p_g_v_v - p_b_c);

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
                        Estimator::MeasVec& z, Estimator::MeasH& H,
                        const Eigen::Matrix3d& cam_K,
                        const Eigen::Vector3d& p_b_c, const quat::Quatd& q_b_c)
{
  meas_dims = 2;
  z.setZero();
  H.setZero();

  // Camera Params
  const double fx = cam_K(0, 0);
  const double fy = cam_K(1, 1);
  const double cx = cam_K(0, 2);
  const double cy = cam_K(1, 2);
  //const double fx = 410.;
  //const double fy = 420.;
  //const double cx = 320.;
  //const double cy = 240.;

  // Constants
  static const Eigen::Vector3d e1(1., 0., 0.);
  static const Eigen::Vector3d e2(0., 1., 0.);
  static const Eigen::Vector3d e3(0., 0., 1.);
  //Eigen::Vector4d q(0.7071, 0., 0., 0.7071);
  //q /= q.norm();
  //quat::Quatd q_b_c(q);
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

  const Eigen::Vector3d p_g_c_c = R_b_c * (R_I_b * p_g_v_v - p_b_c);

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

void landmarkPixelMeasModel(const int& lm_index, const Estimator::StateVec& x,
                            int& meas_dims, Estimator::MeasVec& z,
                            Estimator::MeasH& H, const Eigen::Matrix3d& cam_K,
                            const Eigen::Vector3d& p_b_c,
                            const quat::Quatd& q_b_c)
{
  meas_dims = 2;
  z.setZero();
  H.setZero();

  // Landmarks are 0 indexed
  const int xLM_IDX = Estimator::xGOAL_LM + 3 * lm_index;

  // Camera Params
  const double fx = cam_K(0, 0);
  const double fy = cam_K(1, 1);
  const double cx = cam_K(0, 2);
  const double cy = cam_K(1, 2);
  //const double fx = 410.;
  //const double fy = 420.;
  //const double cx = 320.;
  //const double cy = 240.;

  // Constants
  static const Eigen::Vector3d e1(1., 0., 0.);
  static const Eigen::Vector3d e2(0., 1., 0.);
  static const Eigen::Vector3d e3(0., 0., 1.);
  //Eigen::Vector4d q(0.7071, 0., 0., 0.7071);
  //q /= q.norm();
  //quat::Quatd q_b_c(q);
  const Eigen::Matrix3d R_b_c = q_b_c.R();

  const double phi = x(Estimator::xATT + 0);
  const double theta = x(Estimator::xATT + 1);
  const double psi = x(Estimator::xATT + 2);
  const Eigen::Matrix3d R_I_b = rotmItoB(phi, theta, psi);

  const double theta_g = x(Estimator::xGOAL_ATT);
  const Eigen::Matrix3d R_I_g = rotm3dItoB(theta_g);

  const Eigen::Vector3d p_i_g_g = x.segment<3>(xLM_IDX);
  const Eigen::Vector3d p_i_g_v = R_I_g.transpose() * p_i_g_g;
  //const Eigen::Vector3d p_i_v_v_2d =
      //p_i_g_v.segment<2>(0) + x.segment<2>(Estimator::xGOAL_POS);

  const double rho = x(Estimator::xGOAL_RHO);
  const Eigen::Vector3d p_g_v_v(x(Estimator::xGOAL_POS), x(Estimator::xGOAL_POS + 1), 1. / rho);
  const Eigen::Vector3d p_i_v_v = p_i_g_v + p_g_v_v;
  //std::cout << "x: " << x << std::endl;
  //std::cout << "p_i_v_v: " << p_i_v_v << std::endl;
  //std::cout << "p_i_g_v: " << p_i_g_v << std::endl;
  //std::cout << "p_g_v_v: " << p_g_v_v << std::endl;

  // TODO: this doesn't account for the position offset of a camera
  const Eigen::Vector3d p_i_c_c = R_b_c * (R_I_b * p_i_v_v - p_b_c);

  // const double rho_g = x(Estimator::xGOAL_RHO);
  // Eigen::Vector3d p_g_v_v;
  // p_g_v_v(0) = x(Estimator::xGOAL_POS + 0);
  // p_g_v_v(1) = x(Estimator::xGOAL_POS + 1);
  // p_g_v_v(2) = 1. / rho_g;

  // const Eigen::Vector3d p_g_c_c = R_b_c * R_I_b * p_g_v_v;

  // R_I_b = RotInertial2Body(phi, theta, psi)
  // R_v_g = Rot2DInertial2Body(theta_g)

  // p_i_g_g = np.array([rx_i, ry_i])
  // p_i_g_v = np.matmul(R_v_g.transpose(), p_i_g_g)
  // p_i_v_v_2d = p_i_g_v + np.array([px_g, py_g])

  // p_i_v_v = np.array([p_i_v_v_2d[0], p_i_v_v_2d[1], 1. / rho_i])

  // p_i_c_c = np.matmul(RBC, np.matmul(R_I_b, p_i_v_v) - PCBB)

  // pix_x = FX * (p_i_c_c[0] / p_i_c_c[2]) + CX
  // pix_y = FY * (p_i_c_c[1] / p_i_c_c[2]) + CY

  // Measurement Model
  const double px_hat = fx * (p_i_c_c(0) / p_i_c_c(2)) + cx;
  const double py_hat = fy * (p_i_c_c(1) / p_i_c_c(2)) + cy;
  z(0) = px_hat;
  z(1) = py_hat;

  // Measurement Model Jacobian
  const Eigen::Matrix3d d_R_d_phi = dRIBdPhi(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_theta = dRIBdTheta(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_psi = dRIBdPsi(phi, theta, psi);

  const Eigen::Vector3d RdRdPhip = R_b_c * d_R_d_phi * p_i_v_v;
  const double dpx_dphi =
      (fx * RdRdPhip(0) / p_i_c_c(2)) -
      (fx * RdRdPhip(2) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  const Eigen::Vector3d RdRdThetap = R_b_c * d_R_d_theta * p_i_v_v;
  const double dpx_dtheta =
      (fx * RdRdThetap(0) / p_i_c_c(2)) -
      (fx * RdRdThetap(2) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  const Eigen::Vector3d RdRdPsip = R_b_c * d_R_d_psi * p_i_v_v;
  const double dpx_dpsi =
      (fx * RdRdPsip(0) / p_i_c_c(2)) -
      (fx * RdRdPsip(2) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));

  const Eigen::Vector3d dpx_dp =
      ((fx * e1.transpose() * R_b_c * R_I_b) / p_i_c_c(2)) -
      ((fx * e3.transpose() * R_b_c * R_I_b * p_i_c_c(0)) /
       (p_i_c_c(2) * p_i_c_c(2)));
  const double dpx_drho = -(1. / rho / rho) * dpx_dp(2);

  H.setZero();
  H(0, Estimator::xATT + 0) = dpx_dphi;
  H(0, Estimator::xATT + 1) = dpx_dtheta;
  H(0, Estimator::xATT + 2) = dpx_dpsi;
  H.block<1, 2>(0, Estimator::xGOAL_POS) = dpx_dp.head(2);
  H(0, Estimator::xGOAL_RHO) = dpx_drho;

  const double dpy_dphi =
      (fy * RdRdPhip(1) / p_i_c_c(2)) -
      (fy * RdRdPhip(2) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  const double dpy_dtheta =
      (fy * RdRdThetap(1) / p_i_c_c(2)) -
      (fy * RdRdThetap(2) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  const double dpy_dpsi =
      (fy * RdRdPsip(1) / p_i_c_c(2)) -
      (fy * RdRdPsip(2) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));

  const Eigen::Vector3d dpy_dp =
      ((fy * e2.transpose() * R_b_c * R_I_b) / p_i_c_c(2)) -
      ((fy * e3.transpose() * R_b_c * R_I_b * p_i_c_c(1)) /
       (p_i_c_c(2) * p_i_c_c(2)));
  const double dpy_drho = -(1. / rho / rho) * dpy_dp(2);

  H(1, Estimator::xATT + 0) = dpy_dphi;
  H(1, Estimator::xATT + 1) = dpy_dtheta;
  H(1, Estimator::xATT + 2) = dpy_dpsi;
  H.block<1, 2>(1, Estimator::xGOAL_POS) = dpy_dp.head(2);
  H(1, Estimator::xGOAL_RHO) = dpy_drho;

  // d / d theta_g
  const Eigen::Matrix3d d_R_d_theta_g = dR3DdTheta(theta_g);
  const Vector3d d_theta_p_i_v_v = d_R_d_theta_g.transpose() * p_i_g_g;
  //const Vector3d d_theta_p_i_v_v(d_theta_p_i_v_v_2d(0), d_theta_p_i_v_v_2d(1),
                                 //0.);

  const Eigen::Vector3d RRdRdThetaP = R_b_c * R_I_b * d_theta_p_i_v_v;
  const double dpx_dtheta_g =
      (fx * RRdRdThetaP(0) / p_i_c_c(2)) -
      (fx * RRdRdThetaP(2) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  const double dpy_dtheta_g =
      (fy * RRdRdThetaP(1) / p_i_c_c(2)) -
      (fy * RRdRdThetaP(2) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  H(0, Estimator::xGOAL_ATT) = dpx_dtheta_g;
  H(1, Estimator::xGOAL_ATT) = dpy_dtheta_g;

  // d_rxy_p_i_v_v = np.zeros((3, 2))
  // d_rxy_p_i_v_v[0, 0:2] = R_v_g.transpose()[0, :]
  // d_rxy_p_i_v_v[1, 0:2] = R_v_g.transpose()[1, :]

  // d1drxy = -np.matmul(E3, np.matmul(RBC, np.matmul(R_I_b,
  // d_rxy_p_i_v_v))) * FX * (p_i_c_c[0] / p_i_c_c[2] /
  // p_i_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(R_I_b,
  // d_rxy_p_i_v_v))) / p_i_c_c[2]
  // d2drxy = -np.matmul(E3, np.matmul(RBC, np.matmul(R_I_b,
  // d_rxy_p_i_v_v))) * FY * (p_i_c_c[1] / p_i_c_c[2] /
  // p_i_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(R_I_b,
  // d_rxy_p_i_v_v))) / p_i_c_c[2]

  // jac[0, 17:19] = d1drxy
  // jac[1, 17:19] = d2drxy

  // d / d rxy
  //Eigen::Matrix<double, 3, 2> d_rxy_p_i_v_v;
  //d_rxy_p_i_v_v.block<2, 2>(0, 0) = R_I_g.transpose();
  const Eigen::Matrix3d d_r_p_i_v_v = R_I_g.transpose();
  //d_r_p_i_v_v.block<2, 2>(0, 0) = R_I_g.transpose();

  //const Eigen::Matrix<double, 3, 2> RRdRdrxyp = R_b_c * R_I_b * d_rxy_p_i_v_v;
  const Eigen::Matrix3d RRdRdrp = R_b_c * R_I_b * d_r_p_i_v_v;
  const Eigen::Matrix<double, 1, 3> dpx_dr =
      (fx * RRdRdrp.block<1, 3>(0, 0) / p_i_c_c(2)) -
      (fx * RRdRdrp.block<1, 3>(2, 0) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  const Eigen::Matrix<double, 1, 3> dpy_dr =
      (fy * RRdRdrp.block<1, 3>(1, 0) / p_i_c_c(2)) -
      (fy * RRdRdrp.block<1, 3>(2, 0) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  H.block<1, 3>(0, xLM_IDX) = dpx_dr;
  H.block<1, 3>(1, xLM_IDX) = dpy_dr;
}
