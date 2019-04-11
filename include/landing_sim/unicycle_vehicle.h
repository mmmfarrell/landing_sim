#pragma once

#include <string>
#include <Eigen/Core>
#include <math.h>

#include "multirotor_sim/utils.h"

#include "multirotor_sim/state.h"
#include "multirotor_sim/vehicle_base.h"

void toRot(const double theta, Matrix2d& R)
{
  R << cos(theta), sin(theta),
       -sin(theta), cos(theta);
}

void toRot(const double theta, Matrix3d& R)
{
  R << cos(theta), sin(theta), 0.,
       -sin(theta), cos(theta), 0.,
       0., 0., 1.;
}

namespace  multirotor_sim
{

class UnicycleVehicle : public VehicleBase
{
public:
  enum
  {
    xPOS = 0,
    xVEL = 2,
    xATT = 4,
    xOMEGA = 5,
    xZ = 6
  };

  enum
  {
    SIZE = 5
  };

  typedef Eigen::Matrix<double, xZ, 1> StateVec;
  typedef Eigen::Matrix<double, SIZE, 3> LandmarkVec;

  UnicycleVehicle(std::string filename)
  {
    get_yaml_eigen("x0", filename, x_);
    get_yaml_eigen("landmarks_body_frame", filename, landmarks_body_);
  }

  void step(const double& dt)
  {
    // Euler integration
    const int num_sub_steps = 10;
    const double time_step = dt / num_sub_steps;

    for (int i = 0; i < num_sub_steps; i++)
    {
      const double theta = x_(xATT);
      const double omega = x_(xOMEGA);
      const Vector2d vel_b = x_.block<2, 1>(xVEL, 0);

      Matrix2d R_I_b;
      toRot(theta, R_I_b);

      const Vector2d vel_I = R_I_b.transpose() * vel_b;

      x_.block<2, 1>(xPOS, 0) += time_step * vel_I;
      x_(xATT) += time_step * x_(xOMEGA);
    }
  }


  void landmarkLocations(std::vector<Vector3d>& pts)
  {
    // Grab position and append a zero for altitude
    Vector3d pos_I;
    pos_I.setZero();
    pos_I.block<2, 1>(0, 0) = x_.block<2, 1>(xPOS, 0);

    // Get current rotation matrix
    const double theta = x_(xATT);
    Matrix3d R_I_b;
    toRot(theta, R_I_b);

    // For each landmark point, rotate and translate into inertial frame
    pts.clear();
    for (int i = 0; i < SIZE; i++)
    {
      Vector3d pt_b(landmarks_body_.block<1, 3>(i, 0));
      Vector3d pt_I = R_I_b.transpose() * pt_b + pos_I;
      pts.push_back(pt_I);
    }
  }

  StateVec x_;
  LandmarkVec landmarks_body_;
};

}
