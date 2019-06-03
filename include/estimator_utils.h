#pragma once

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
