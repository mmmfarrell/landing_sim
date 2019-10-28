#pragma once

#include <set>
#include <string>

#include <Eigen/Core>

#include <geometry/xform.h>


namespace meas
{

using Vector1d = Eigen::Matrix<double, 1, 1>;
using Matrix1d = Eigen::Matrix<double, 1, 1>;

struct Base
{
    Base();
    virtual ~Base() = default;
    enum
    {
        BASE,
        GNSS,
        IMU,
        BARO,
        ALT,
        MOCAP,
        ARUCO,
        ZERO_VEL
    };
    double t;
    int type;
    bool handled;
    std::string Type() const;
};

typedef std::multiset<meas::Base*, std::function<bool(const meas::Base*, const meas::Base*)>> MeasSet;
bool basecmp(const Base *a, const Base *b);

struct Gnss : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Gnss(double _t, const Vector6d& _z, const Matrix6d& _R);
    Vector6d z;
    Eigen::Map<Eigen::Vector3d> p;
    Eigen::Map<Eigen::Vector3d> v;
    Matrix6d R;
};

struct Imu : public Base
{
    enum
    {
        A = 0,
        W = 3
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Imu(double _t, const Vector6d& _z, const Matrix6d& _R);
    Vector6d z;
    Matrix6d R;
};

struct Baro : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Baro(double _t, const double& _z, const double& _R, const double& _temp);
    Eigen::Matrix<double, 1, 1> z;
    Eigen::Matrix<double, 1, 1> R;
    double temp;
};

struct Alt : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Alt(double _t, const Vector1d& _z, const Matrix1d& _R);
    // Eigen::Matrix<double, 1, 1> z;
    // Eigen::Matrix<double, 1, 1> R;
    Vector1d z;
    Matrix1d R;
};

struct Mocap : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Mocap(double _t, const xform::Xformd& _z, const Matrix6d& _R);
    xform::Xformd z;
    Matrix6d R;
};

struct Aruco : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Aruco(double _t, const Eigen::Vector3d& _z, const Eigen::Matrix3d& _R,
          const quat::Quatd& _q_c2a, const Matrix1d& _yaw_R);
    Eigen::Vector3d z;
    Eigen::Matrix3d R;
    quat::Quatd q_c2a;
    Matrix1d yaw_R;
};

struct ZeroVel: public Base
{
    ZeroVel(double _t);
};

}

