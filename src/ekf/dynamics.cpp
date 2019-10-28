#include "ekf/ekf.h"

using namespace Eigen;
using namespace xform;
using namespace quat;
using namespace std;

#define T transpose()
namespace ekf
{


void EKF::dynamics(const State &x, const Vector6d& u, ErrorState &dx, bool calc_jac)
{
    Vector3d accel = u.head<3>() - x.ba;
    Vector3d omega = u.tail<3>() - x.bg;

    dx.p = x.q.rota(x.v);
    dx.q = omega;
    dx.v = accel + x.q.rotp(gravity) - omega.cross(x.v);
    dx.ba.setZero();
    dx.bg.setZero();
    dx.bb = 0.;
    dx.ref = 0.;

    if (goal_initialized_)
    {
      const Eigen::Matrix2d R_I2g = rotm2dItoB(x.gatt);
      const Eigen::Vector2d goal_vel_I_2d = R_I2g.transpose() * x.gv;
      const Eigen::Vector3d goal_vel_I(goal_vel_I_2d(0), goal_vel_I_2d(1), 0.);

      dx.gp = goal_vel_I - x.q.rota(x.v);
      // dx.gp.head(2) = goal_vel_I - x.q.rota(x.v).head<2>();
      // dx.gp(2) = 0.;
      dx.gv.setZero();
      dx.gatt = x.gw;
      dx.gw = 0.;
      dx.lms.setZero();
    }
    else
    {
      dx.gp.setZero();
      dx.gv.setZero();
      dx.gatt = 0.;
      dx.gw = 0.;
      dx.lms.setZero();
    }

    CHECK_NAN(dx.arr);
    if (calc_jac)
    {
        Matrix3d R = x.q.R();
        typedef ErrorState DX;
        typedef meas::Imu U;

        A_.setZero();
        B_.setZero();
        A_.block<3,3>(DX::DP, DX::DQ) = -R.T * skew(x.v);
        A_.block<3,3>(DX::DP, DX::DV) = R.T;

        A_.block<3,3>(DX::DQ, DX::DQ) = -skew(omega);
        A_.block<3,3>(DX::DQ, DX::DBG) = -I_3x3;
        B_.block<3,3>(DX::DQ, U::W) = I_3x3;

        A_.block<3,3>(DX::DV, DX::DV) = -skew(omega);
        A_.block<3,3>(DX::DV, DX::DQ) = skew(x.q.rotp(gravity));
        A_.block<3,3>(DX::DV, DX::DBA) = -I_3x3;
        A_.block<3,3>(DX::DV, DX::DBG) = -skew(x.v);
        B_.block<3,3>(DX::DV, U::A) = I_3x3;
        B_.block<3,3>(DX::DV, U::W) = skew(x.v);

        if (goal_initialized_)
        {
          // A_.block<2,3>(DX::DGP, DX::DQ) = (R.T * skew(x.v)).topRows(2);
          // A_.block<2,3>(DX::DGP, DX::DV) = (-R.T).topRows(2);
          A_.block<3,3>(DX::DGP, DX::DQ) = R.T * skew(x.v);
          A_.block<3,3>(DX::DGP, DX::DV) = -R.T;

          const Eigen::Matrix2d R_I2g = rotm2dItoB(x.gatt);
          const Eigen::Matrix2d dR_v2g_dTheta = dR2DdTheta(x.gatt);
          A_.block<2, 2>(DX::DGP, DX::DGV) = R_I2g.transpose();
          A_.block<2, 1>(DX::DGP, DX::DGATT) = dR_v2g_dTheta.transpose() * x.gv;

          A_(DX::DGATT, DX::DGW) =  1.;
        }

        CHECK_NAN(A_); CHECK_NAN(B_);
    }
}

Eigen::Matrix2d EKF::rotm2dItoB(const double theta)
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

Eigen::Matrix2d EKF::dR2DdTheta(const double theta)
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

void EKF::errorStateDynamics(const State& xhat, const ErrorState& xt, const Vector6d& u,
                             const Vector6d& eta, ErrorState& dxdot)
{
    auto eta_a = eta.head<3>();
    auto eta_w = eta.tail<3>();
    auto z_a = u.head<3>();
    auto z_w = u.tail<3>();

    State x = xhat + xt;
    Vector3d a = z_a - x.ba + eta_a;
    Vector3d ahat = z_a - xhat.ba;
    Vector3d w = z_w - x.bg + eta_w;
    Vector3d what = z_w - xhat.bg;
    dxdot.arr.setZero();
    dxdot.p = x.q.rota(x.v) - xhat.q.rota(xhat.v);
    dxdot.q = w - x.q.rotp(xhat.q.rota(what));
    dxdot.v = x.q.rotp(gravity) + a - w.cross(x.v) - (xhat.q.rotp(gravity) + ahat - what.cross(xhat.v));
}
}
