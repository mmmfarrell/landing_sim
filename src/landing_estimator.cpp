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
  // Init EKF State
  xhat_.setZero();
  xhat_(xMU) = 0.1;
  xhat_(xGOAL_RHO) = 0.2;

  P_.setIdentity();
  P_(xMU, xMU) = 0.;
  P_(xGOAL_RHO, xGOAL_RHO) = 0.01;
  //P_.setZero();

  Qx_.setIdentity();
  Qx_ *= 0.00001;
  Qx_(xMU, xMU) = 0.;
  Qx_(xGOAL_RHO, xGOAL_RHO) = 0.0000001;

  Qu_.setIdentity();
  Qu_(uAZ, uAZ) = 0.2 * 0.2;
  Qu_.block<3, 3>(uOMEGA, uOMEGA) *= 0.1 * 0.1;

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

  //// update with ax, ay
  //const double imu_dims = 2;
  //const double xhat_mu = xhat_(xMU);
  //const Eigen::Vector2d xhat_accel = -xhat_mu * xhat_.segment<2>(xVEL);
  //z_resid_.head(imu_dims) = z.head(2) - xhat_accel;
  ////z_R_.topLeftCorner(imu_dims, imu_dims) = 0.2 * 0.2 * Eigen::Matrix2d::Identity();
  //z_R_.topLeftCorner(imu_dims, imu_dims) = R.topLeftCorner(imu_dims, imu_dims);

  //H_.setZero();
  //H_(0, xVEL + 0) = -xhat_mu;
  //H_(0, xMU) = -xhat_(xVEL + 0);
  //H_(1, xVEL + 1) = -xhat_mu;
  //H_(1, xMU) = -xhat_(xVEL + 1);

  //update(imu_dims, z_resid_, z_R_, H_);
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
  //const double pos_dims = 3;
  //const Vector3d mocap_pos = z.t();
  //z_resid_.head(pos_dims) = mocap_pos - xhat_.segment<3>(xPOS);
  //z_R_.topLeftCorner(pos_dims, pos_dims) = 0.001 * Eigen::Matrix3d::Identity();

  //H_.setZero();
  //H_.block<3, 3>(0, xPOS) = Eigen::Matrix3d::Identity();

  //update(pos_dims, z_resid_, z_R_, H_);

  // Update atttitude
  const double att_dims = 3;
  const Vector3d mocap_euler = z.q().euler();
  z_resid_.head(att_dims) = mocap_euler - xhat_.segment<3>(xATT);
  //z_R_.topLeftCorner(att_dims, att_dims) = 0.001 * Eigen::Matrix3d::Identity();
  z_R_.topLeftCorner(att_dims, att_dims) = R.block<3, 3>(3, 3);

  H_.setZero();
  H_.block<3, 3>(0, xATT) = Eigen::Matrix3d::Identity();

  update(att_dims, z_resid_, z_R_, H_);

  //// Update atttitude no yaw
  //const double att_dims = 2;
  //const Vector3d mocap_euler = z.q().euler();
  //z_resid_.head(att_dims) = mocap_euler.head(2) - xhat_.segment<2>(xATT);
  //z_R_.topLeftCorner(att_dims, att_dims) = 0.001 * Eigen::Matrix2d::Identity();

  //H_.setZero();
  //H_.block<2, 2>(0, xATT) = Eigen::Matrix2d::Identity();

  //update(att_dims, z_resid_, z_R_, H_);
}

void Estimator::velocityCallback(const double& t, const Vector3d& vel_b,
                                 const Matrix3d& R)
{
}

void Estimator::simpleCamCallback(const double& t, const ImageFeat& z,
                                  const Matrix2d& R_pix,
                                  const Matrix1d& R_depth)
{
  // Update goal depth
  const int depth_dims = 1;
  const double zhat_depth = 1./ xhat_(xGOAL_RHO);
  z_resid_(0) = z.depths[0] - zhat_depth;
  z_R_.topLeftCorner(depth_dims, depth_dims) = 0.1 * Matrix1d::Identity();
  H_.setZero();
  H_(0, xGOAL_RHO) = - zhat_depth * zhat_depth;
  update(depth_dims, z_resid_, z_R_, H_);
  //std::cout << std::endl;
  //std::cout << "Goal pix px: " << z.pixs[0](0) << " py: " << z.pixs[0](1) << std::endl;

  int pix_dims = 0;
  MeasVec pix_zhat;
  goalPixelMeasModel(xhat_, pix_dims, pix_zhat, H_);

  z_resid_.head(pix_dims) = pix_zhat.head(pix_dims) - z.pixs[0];
  z_R_.topLeftCorner(pix_dims, pix_dims) = 4.0 * Eigen::Matrix2d::Identity();
  //update(pix_dims, z_resid_, z_R_, H_);
}

void Estimator::gnssCallback(const double& t, const Vector6d& z,
                             const Matrix6d& R)
{
  // update position
  const double pos_dims = 3;
  const Vector3d gnss_pos = z.head(pos_dims);
  z_resid_.head(pos_dims) = gnss_pos - xhat_.segment<3>(xPOS);
  //z_R_.topLeftCorner(pos_dims, pos_dims) = 0.1 * Eigen::Matrix3d::Identity();
  z_R_.topLeftCorner(pos_dims, pos_dims) = R.topLeftCorner(pos_dims, pos_dims);

  H_.setZero();
  H_.block<3, 3>(0, xPOS) = Eigen::Matrix3d::Identity();

  update(pos_dims, z_resid_, z_R_, H_);

  // update velocity
  const double vel_dims = 3;
  const Eigen::Vector3d gnss_vel_I = z.segment<3>(3);

  const double phi = xhat_(xATT + 0);
  const double theta = xhat_(xATT + 1);
  const double psi = xhat_(xATT + 2);
  const Eigen::Matrix3d R_I_b = rotmItoB(phi, theta, psi);
  const Eigen::Vector3d xhat_vel_b = xhat_.segment<3>(xVEL);
  const Eigen::Vector3d zhat_vel_I = R_I_b.transpose() * xhat_vel_b;

  z_resid_.head(vel_dims) = gnss_vel_I - zhat_vel_I;
  z_R_.topLeftCorner(vel_dims, vel_dims) = R.block<3, 3>(3, 3);
  //z_R_.topLeftCorner(vel_dims, vel_dims) = 0.1 * Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d d_R_d_phi = dRIBdPhi(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_theta = dRIBdTheta(phi, theta, psi);
  const Eigen::Matrix3d d_R_d_psi = dRIBdPsi(phi, theta, psi);
  H_.setZero();
  H_.block<3, 1>(0, xATT + 0) = d_R_d_phi * xhat_vel_b;
  H_.block<3, 1>(0, xATT + 1) = d_R_d_theta * xhat_vel_b;
  H_.block<3, 1>(0, xATT + 2) = d_R_d_psi * xhat_vel_b;
  H_.block<3, 3>(0, xVEL) = R_I_b.transpose();

  update(vel_dims, z_resid_, z_R_, H_);
}

void Estimator::propagate(const double& dt, const InputVec& u_in)
{
  dynamics(xhat_, u_, xdot_, A_, G_);

  xhat_ += xdot_ * dt;

  //A_ = I_ + A_ * dt;
  //P_ = A_ * P_ * A_.transpose() + Q_;

  // FROM VIEKF
  G_ = (I_ + A_*dt/2.0 + A_*A_*dt*dt/6.0)*G_*dt;
  A_ = I_ + A_*dt + A_*A_*dt*dt/2.0;
  P_ = A_ * P_ * A_.transpose() + G_ * Qu_ * G_.transpose() + Qx_;
}

void Estimator::update(const double dims, const MeasVec& residual,
                       const MeasMat& R, const MeasH& H)
{
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
  //A(xPOS + 0, xATT + 0) =
      //(cp * st * cpsi + sp * spsi) * v + (-sp * st * cpsi + cp * spsi) * w;
  //A(xPOS + 0, xATT + 1) =
      //(-st * cpsi) * u + (sp * ct * cpsi) * v + (cp * ct * cpsi) * w;
  //A(xPOS + 0, xATT + 2) = (-ct * spsi) * u + (-sp * st * spsi - cp * cpsi) * v +
                          //(-cp * st * spsi + sp * cpsi) * w;

  //A(xPOS + 1, xATT + 0) =
      //(cp * st * spsi - sp * cpsi) * v + (-sp * st * spsi - cp * cpsi) * w;
  //A(xPOS + 1, xATT + 1) =
      //(-st * spsi) * u + (sp * ct * spsi) * v + (cp * ct * spsi) * w;
  //A(xPOS + 1, xATT + 2) = (ct * cpsi) * u + (sp * st * cpsi - cp * spsi) * v +
                          //(cp * st * cpsi + sp * spsi) * w;

  //A(xPOS + 2, xATT + 0) = (cp * ct) * v + (-sp * ct) * w;
  //A(xPOS + 2, xATT + 1) = (-ct) * u + (-sp * st) * v + (-cp * st) * w;
  //A(xPOS + 2, xATT + 2) = 0.;
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
  xdot.segment<2>(xGOAL_POS) = R_v_g.transpose() * vel_g - I_2x3 * R_I_b.transpose() * vel_b;
  xdot(xGOAL_RHO) = rho_g * rho_g * e3.transpose() * R_I_b.transpose() * vel_b;
  //xdot.segment<2>(xGOAL_VEL) = 0.;
  xdot(xGOAL_ATT) = omega_g;
  //xdot(xGOAL_OMEGA) = 0.;

  // Landing goal jacobian
  // d goal / d UAV
  A.block<2, 1>(xGOAL_POS, xATT + 0) = -I_2x3 * d_R_d_phi.transpose() * vel_b;
  A.block<2, 1>(xGOAL_POS, xATT + 1) = -I_2x3 * d_R_d_theta.transpose() * vel_b;
  A.block<2, 1>(xGOAL_POS, xATT + 2) = -I_2x3 * d_R_d_psi.transpose() * vel_b;
  A.block<2, 3>(xGOAL_POS, xVEL) = -I_2x3 * R_I_b.transpose();

  A(xGOAL_RHO, xATT + 0) = rho_g * rho_g * e3.transpose() * d_R_d_phi.transpose() * vel_b;
  A(xGOAL_RHO, xATT + 1) = rho_g * rho_g * e3.transpose() * d_R_d_theta.transpose() * vel_b;
  A(xGOAL_RHO, xATT + 2) = rho_g * rho_g * e3.transpose() * d_R_d_psi.transpose() * vel_b;
  A.block<1, 3>(xGOAL_RHO, xVEL) = rho_g * rho_g * e3.transpose() * R_I_b.transpose();

  // d goal / d goal
  A.block<2, 2>(xGOAL_POS, xGOAL_VEL) = R_v_g.transpose();
  A.block<2, 1>(xGOAL_POS, xGOAL_ATT) = dR_v_g_dTheta.transpose() * vel_g;

  A(xGOAL_RHO, xGOAL_RHO) = 2. * rho_g * e3.transpose() * R_I_b.transpose() * vel_b;

  A(xGOAL_ATT, xGOAL_OMEGA) = 1.;


  // Landmark dynamics
  for (unsigned int i = 0; i < MAXLANDMARKS; i++)
  {
    const unsigned int xRHOI = xGOAL_LM + 2 + 3 * i;
    const double rho_i = x(xRHOI);

    // dynamics
    xdot(xRHOI) = rho_i * rho_i * e3.transpose() * R_I_b.transpose() * vel_b;

    // jacobian
    // d goal / d UAV
    A(xRHOI, xATT + 0) = rho_i * rho_i * e3.transpose() * d_R_d_phi.transpose() * vel_b;
    A(xRHOI, xATT + 1) = rho_i * rho_i * e3.transpose() * d_R_d_theta.transpose() * vel_b;
    A(xRHOI, xATT + 2) = rho_i * rho_i * e3.transpose() * d_R_d_psi.transpose() * vel_b;
    A.block<1, 3>(xRHOI, xVEL) = rho_i * rho_i * e3.transpose() * R_I_b.transpose();

    // d goal / d goal
    A(xRHOI, xRHOI) = 2. * rho_i * e3.transpose() * R_I_b.transpose() * vel_b;
  }
}
