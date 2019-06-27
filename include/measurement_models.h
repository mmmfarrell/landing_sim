#ifndef MEASUREMENT_MODELS_H
#define MEASUREMENT_MODELS_H

#include "landing_estimator.h"
#include "estimator_utils.h"

void goalDepthMeasModel(const Estimator::StateVec& x, int& meas_dims,
                        Estimator::MeasVec& z, Estimator::MeasH& H);
void goalPixelMeasModel(const Estimator::StateVec& x, int& meas_dims,
                        Estimator::MeasVec& z, Estimator::MeasH& H);
void landmarkPixelMeasModel(const Estimator::StateVec& x, int& meas_dims,
                        Estimator::MeasVec& z, Estimator::MeasH& H);

#endif /* MEASUREMENT_MODELS_H */
