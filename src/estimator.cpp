#include <iostream>

#include "landing_sim/utils.h"
#include "multirotor_sim/utils.h"

#include "estimator.h"

using namespace Eigen;
using namespace multirotor_sim;

Estimator::Estimator(std::string filename)
{
  get_yaml_node("draw_feature_img", filename, draw_feats_);
  get_yaml_eigen("x0", filename, xhat_);

  StateVec P_diag;
  get_yaml_eigen("P0", filename, P_diag);
  P_ = P_diag.asDiagonal();

  last_time_ = -1;
}

Estimator::~Estimator()
{
}

void Estimator::mocapCallback(const double& t, const Xformd& z, const Matrix6d& R)
{
}

void Estimator::simpleCamCallback(const double& t, const ImageFeat& z,
                       const Matrix2d& R_pix, const Matrix1d& R_depth)
{
  if (draw_feats_)
    drawImageFeatures(z.pixs);

  if (last_time_ < 0)
  {
    last_time_ = t;
    return;
  }

  const double dt = t - last_time_;
  propagate(dt);

  // updateGoal
  // updateGoalDepth
  // updateLandmarks
}

void Estimator::propagate(const double& dt)
{

}

void Estimator::updateGoal(const Vector2d& goal_pix)
{

}

void Estimator::updateGoalDepth(const double& goal_depth)
{

}

void Estimator::updateLandmark(const int& id, const Vector2d& lm_pix)
{

}
