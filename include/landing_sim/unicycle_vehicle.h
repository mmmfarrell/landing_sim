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
    // SIZE = 4
    SIZE = 20
  };

  typedef Eigen::Matrix<double, xZ, 1> StateVec;
  typedef Eigen::Matrix<double, SIZE, 3> LandmarkVec;
  typedef Eigen::Matrix<double, SIZE, 1> IdVec;

  UnicycleVehicle(std::string filename)
  {
    get_yaml_eigen("x0", filename, x_);
    get_yaml_eigen("aruco_body_frame", filename, aruco_body_);

    get_yaml_node("omega_walk_std", filename, omega_walk_std_);
    get_yaml_node("vel_walk_std", filename, vel_walk_std_);

    get_yaml_node("landmark_disappear_prob", filename, lm_disappear_prob_);

    get_yaml_node("seed", filename, seed_);
    if (seed_ == 0)
    {
      seed_ = std::chrono::system_clock::now().time_since_epoch().count();
    }
    rng_.seed(seed_);
    srand(seed_);

    lm_id_counter_ = 0;

    double landmarks_xy_min;
    double landmarks_xy_max;
    double landmarks_z_min;
    double landmarks_z_max;
    get_yaml_node("landmarks_xy_min", filename, landmarks_xy_min);
    get_yaml_node("landmarks_xy_max", filename, landmarks_xy_max);
    get_yaml_node("landmarks_z_min", filename, landmarks_z_min);
    get_yaml_node("landmarks_z_max", filename, landmarks_z_max);

    lm_xy_dist_ = std::uniform_real_distribution<double>(landmarks_xy_min,
                                                         landmarks_xy_max);
    lm_z_dist_ = std::uniform_real_distribution<double>(landmarks_z_min,
                                                        landmarks_z_max);

    initLandmarks(filename);

    uniform_ = std::uniform_real_distribution<double>(0., 1.);
  }

  void initLandmarks(std::string filename)
  {
    for (int i = 0; i < SIZE; ++i)
    {
      lm_id_counter_++;
      landmark_ids_(i) = lm_id_counter_;
      landmarks_body_(i, 0) = lm_xy_dist_(rng_);
      landmarks_body_(i, 1) = lm_xy_dist_(rng_);
      landmarks_body_(i, 2) = lm_z_dist_(rng_);
    }
  }

  void updateLandmarks()
  {
    LandmarkVec new_landmarks;
    IdVec new_ids;

    int new_idx = 0;
    for (int i = 0; i < SIZE; ++i)
    {
      const double rand_num = uniform_(rng_);
      // get random number from uniform(0., 1.)
      // if random number is less than prob, landmark dissappears
      if (rand_num < lm_disappear_prob_)
      {
        // landmark disappears, go to next
        // std::cout << "Unicycle lm disappear # " << landmark_ids_(i) <<
        // std::endl;
        continue;
      }

      // If landmark is still around, add it to the new landmarks
      new_ids(new_idx) = landmark_ids_(i);
      new_landmarks.block<1, 3>(new_idx, 0) = landmarks_body_.block<1, 3>(i, 0);
      new_idx++;
    }

    while (new_idx < SIZE)
    {
      lm_id_counter_++;
      new_ids(new_idx) = lm_id_counter_;
      new_landmarks(new_idx, 0) = lm_xy_dist_(rng_);
      new_landmarks(new_idx, 1) = lm_xy_dist_(rng_);
      new_landmarks(new_idx, 2) = lm_z_dist_(rng_);
      new_idx++;
    }

    landmark_ids_ = new_ids;
    landmarks_body_ = new_landmarks;
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

  void landmarkLocations(std::vector<int>& ids, std::vector<Vector3d>& pts)
  {
    updateLandmarks();

    // Grab position and append a zero for altitude
    Vector3d pos_I;
    pos_I.setZero();
    pos_I.block<2, 1>(0, 0) = x_.block<2, 1>(xPOS, 0);

    // Get current rotation matrix
    const double theta = x_(xATT);
    Matrix3d R_I_b;
    toRot(theta, R_I_b);

    // For each landmark point, rotate and translate into inertial frame
    ids.clear();
    pts.clear();
    for (int i = 0; i < SIZE; i++)
    {
      Vector3d pt_b(landmarks_body_.block<1, 3>(i, 0));
      Vector3d pt_I = R_I_b.transpose() * pt_b + pos_I;
      pts.push_back(pt_I);

      ids.push_back(landmark_ids_(i));
    }
  }

  Vector2d getPosition()
  {
    return x_.block<2, 1>(0, 0);
  }

  StateVec x_;
  Vector3d aruco_body_;
  LandmarkVec landmarks_body_;
  IdVec landmark_ids_;

  int lm_id_counter_;
  double lm_disappear_prob_;
  std::uniform_real_distribution<double> lm_xy_dist_;
  std::uniform_real_distribution<double> lm_z_dist_;

  double omega_walk_std_;
  double vel_walk_std_;

  uint64_t seed_;
  std::default_random_engine rng_;
  std::normal_distribution<double> normal_;
  std::uniform_real_distribution<double> uniform_;
};

}  // namespace multirotor_sim
