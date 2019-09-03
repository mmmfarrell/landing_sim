#ifndef ESTIMATOR_UTILS_H
#define ESTIMATOR_UTILS_H

#include <Eigen/Dense>

Eigen::Matrix3d rotmItoB(const double phi, const double theta,
                         const double psi);
Eigen::Matrix3d dRIBdPhi(const double phi, const double theta,
                         const double psi);
Eigen::Matrix3d dRIBdTheta(const double phi, const double theta,
                           const double psi);
Eigen::Matrix3d dRIBdPsi(const double phi, const double theta,
                         const double psi);

Eigen::Matrix2d rotm2dItoB(const double theta);
Eigen::Matrix2d dR2DdTheta(const double theta);
Eigen::Matrix3d rotm3dItoB(const double theta);
Eigen::Matrix3d dR3DdTheta(const double theta);

Eigen::Matrix3d wMat(const double phi, const double theta, const double psi);

#endif /* ifndef ESTIMATOR_UTILS_H */
