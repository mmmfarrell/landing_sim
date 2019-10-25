#pragma once

#include "ekf/ekf.h"
// #include "ekf/image_feat.h"

#include <mutex>
#include <deque>
#include <vector>

// #include <ros/ros.h>
// #include <ros/package.h>
// #include <sensor_msgs/Imu.h>
// #include <sensor_msgs/Range.h>
// #include <nav_msgs/Odometry.h>
// // #include <rosflight_msgs/Barometer.h>
// #include <sensor_msgs/FluidPressure.h>
// #include <rosflight_msgs/Status.h>
// #include <rosflight_msgs/GNSS.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <geometry_msgs/TransformStamped.h>
// #include <geometry_msgs/Vector3Stamped.h>
// #include <std_msgs/Bool.h>
// #include <feature_tracker/ImageFeatures.h>

// #ifdef UBLOX
// #include "ublox/PosVelEcef.h"
// #endif

// #ifdef INERTIAL_SENSE
// #include "inertial_sense/GPS.h"
// #endif
#include "multirotor_sim/estimator_base.h"
#include "multirotor_sim/state.h"

using namespace Eigen;
using namespace multirotor_sim;

class EKF_SIM : public multirotor_sim::EstimatorBase
{
public:

  EKF_SIM();
  ~EKF_SIM();
  void init(const std::string& param_file);
  // void initROS();

  ekf::State getEstimate() { return ekf_.x(); }
  ekf::dxVec getCovariance() { return ekf_.P().diagonal(); }

  // void imuCallback(const sensor_msgs::ImuConstPtr& msg);
  void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R);
  // void baroCallback(const sensor_msgs::FluidPressureConstPtr& msg);
  // void rangeCallback(const sensor_msgs::RangeConstPtr& msg);
  // void poseCallback(const geometry_msgs::PoseStampedConstPtr &msg);
  void mocapCallback(const double& t, const Xformd& z, const Matrix6d& R);
  // void odomCallback(const nav_msgs::OdometryConstPtr &msg);
  // void gnssCallback(const rosflight_msgs::GNSSConstPtr& msg);
  // void mocapCallback(const ros::Time& time, const xform::Xformd &z);
  // void statusCallback(const rosflight_msgs::StatusConstPtr& msg);

  // void arucoCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  // void landmarksCallback(const feature_tracker::ImageFeaturesConstPtr& msg);
    void arucoCallback(const double& t, const xform::Xformd& x_c2a_meas, const Matrix6d& aruco_R);
    void landmarksCallback(const double& t, const ImageFeat& z, const Matrix2d& R_pix);

    void gnssCallback(const double& t, const Vector6d& z, const Matrix6d& R) {}

// #ifdef UBLOX
  // void gnssCallbackUblox(const ublox::PosVelEcefConstPtr& msg);
// #endif

// #ifdef INERTIAL_SENSE
  // void gnssCallbackInertialSense(const inertial_sense::GPSConstPtr& msg);
// #endif

  
private:
  ekf::EKF ekf_;


  std::mutex ekf_mtx_;

  bool imu_init_ = false;
  bool truth_init_ = false;

  bool use_odom_;
  bool use_pose_;

  bool is_flying_ = false;
  bool armed_ = false;
  // ros::Time time_took_off_;
  // ros::Time start_time_;
  double start_time_ = 0.;

  Vector6d imu_;
  ImageFeat lms_meas_;
  
  Matrix6d imu_R_;
  Matrix6d mocap_R_;
  double baro_R_;
  double range_R_;
  Eigen::Matrix3d aruco_R_;
  Matrix1d aruco_yaw_R_;

  bool manual_gps_noise_;
  double gps_horizontal_stdev_;
  double gps_vertical_stdev_;
  double gps_speed_stdev_;

  // void publishEstimates(const sensor_msgs::ImuConstPtr &msg);
};





