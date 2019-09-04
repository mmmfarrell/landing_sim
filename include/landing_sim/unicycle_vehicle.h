#pragma once

#include <string>
#include <Eigen/Core>
#include <math.h>
#include <random>

#include "multirotor_sim/utils.h"

#include "multirotor_sim/state.h"
#include "multirotor_sim/vehicle_base.h"

void toRot(const double theta, Matrix2d& R)
{
  R << cos(theta), sin(theta), -sin(theta), cos(theta);
}

void toRot(const double theta, Matrix3d& R)
{
  R << cos(theta), sin(theta), 0., -sin(theta), cos(theta), 0., 0., 0., 1.;
}

namespace multirotor_sim
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
    SIZE = 20
  };

  typedef Eigen::Matrix<double, xZ, 1> StateVec;
  typedef Eigen::Matrix<double, SIZE, 3> LandmarkVec;

  UnicycleVehicle(std::string filename)
  {
    get_yaml_eigen("x0", filename, x_);
    get_yaml_eigen("aruco_body_frame", filename, aruco_body_);

    get_yaml_node("omega_walk_std", filename, omega_walk_std_);
    get_yaml_node("vel_walk_std", filename, vel_walk_std_);

    get_yaml_node("seed", filename, seed_);
    if (seed_ == 0)
    {
      seed_ = std::chrono::system_clock::now().time_since_epoch().count();
    }
    rng_.seed(seed_);
    srand(seed_);

    init_landmarks(filename);
  }

  void init_landmarks(std::string filename)
  {
    double landmarks_xy_min;
    double landmarks_xy_max;
    double landmarks_z_min;
    double landmarks_z_max;
    get_yaml_node("landmarks_xy_min", filename, landmarks_xy_min);
    get_yaml_node("landmarks_xy_max", filename, landmarks_xy_max);
    get_yaml_node("landmarks_z_min", filename, landmarks_z_min);
    get_yaml_node("landmarks_z_max", filename, landmarks_z_max);

    std::uniform_real_distribution<double> xy_dist(landmarks_xy_min,
                                                   landmarks_xy_max);
    std::uniform_real_distribution<double> z_dist(landmarks_z_min,
                                                  landmarks_z_max);

    for (int i = 0; i < SIZE; ++i)
    {
      landmarks_body_(i, 0) = xy_dist(rng_);
      landmarks_body_(i, 1) = xy_dist(rng_);
      landmarks_body_(i, 2) = z_dist(rng_);
    }
  }

  void step(const double& dt)
  {
    // Euler integration
    const int num_sub_steps = 10;
    const double time_step = dt / num_sub_steps;

    for (int i = 0; i < num_sub_steps; i++)
    {
      const double theta = x_(xATT);
      Matrix2d R_I_b;
      toRot(theta, R_I_b);

      const double omega = x_(xOMEGA);
      const Vector2d vel_b = x_.block<2, 1>(xVEL, 0);

      const Vector2d vel_I = R_I_b.transpose() * vel_b;

      // dynamics
      x_.block<2, 1>(xPOS, 0) += time_step * vel_I;
      x_(xATT) += time_step * omega;

      // random walk
      x_(xVEL + 0) += time_step * vel_walk_std_ * normal_(rng_);
      x_(xVEL + 1) += time_step * vel_walk_std_ * normal_(rng_);
      x_(xOMEGA) += time_step * omega_walk_std_ * normal_(rng_);
    }
  }

  void arucoLocation(Vector3d& pt)
  {
    // Grab position and append a zero for altitude
    Vector3d pos_I;
    pos_I.setZero();
    pos_I.block<2, 1>(0, 0) = x_.block<2, 1>(xPOS, 0);

    // Get current rotation matrix
    const double theta = x_(xATT);
    Matrix3d R_I_b;
    toRot(theta, R_I_b);

    // return the aruco point rotated into the inertial frame
    pt = R_I_b.transpose() * aruco_body_ + pos_I;
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

  Vector2d getPosition()
  {
    return x_.block<2, 1>(0, 0);
  }

  StateVec x_;
  Vector3d aruco_body_;
  LandmarkVec landmarks_body_;

  double omega_walk_std_;
  double vel_walk_std_;

  uint64_t seed_;
  std::default_random_engine rng_;
  std::normal_distribution<double> normal_;
};

}  // namespace multirotor_sim
