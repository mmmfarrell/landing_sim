#ifndef MEASUREMENT_MODELS_H
#define MEASUREMENT_MODELS_H

#include <Eigen/Core>
#include "geometry/quat.h"

#include "landing_estimator.h"
#include "estimator_utils.h"

void goalDepthMeasModel(const Estimator::StateVec& x, int& meas_dims,
                        Estimator::MeasVec& z, Estimator::MeasH& H,
                        const Eigen::Vector3d& p_b_c, const quat::Quatd& q_b_c);

void goalPixelMeasModel(const Estimator::StateVec& x, int& meas_dims,
                        Estimator::MeasVec& z, Estimator::MeasH& H,
                        const Eigen::Matrix3d& cam_K,
                        const Eigen::Vector3d& p_b_c, const quat::Quatd& q_b_c);
void landmarkPixelMeasModel(const int& lm_index, const Estimator::StateVec& x,
                            int& meas_dims, Estimator::MeasVec& z,
                            Estimator::MeasH& H, const Eigen::Matrix3d& cam_K,
                            const Eigen::Vector3d& p_b_c,
                            const quat::Quatd& q_b_c);

#endif /* MEASUREMENT_MODELS_H */
