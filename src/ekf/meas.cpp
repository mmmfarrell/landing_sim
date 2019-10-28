#include "ekf/meas.h"

namespace meas
{

Base::Base()
{
    type = BASE;
    handled = false;
}

std::string Base::Type() const
{
    switch (type)
    {
    case BASE:
        return "Base";
        break;
    case GNSS:
        return "Gnss";
        break;
    case IMU:
        return "Imu";
        break;
    case BARO:
        return "Baro";
        break;
    case ALT:
        return "Alt";
        break;
    case MOCAP:
        return "Mocap";
        break;
    case ARUCO:
        return "Aruco";
        break;
    case ZERO_VEL:
        return "ZeroVel";
        break;
    }
}

bool basecmp(const Base* a, const Base* b)
{
    return a->t < b->t;
}



Imu::Imu(double _t, const Vector6d &_z, const Matrix6d &_R)
{
    t = _t;
    z = _z;
    R = _R;
    type = IMU;
}

Baro::Baro(double _t, const double &_z, const double &_R, const double& _temp)
{
    t = _t;
    z(0) = _z;
    R(0) = _R;
    temp = _temp;
    type = BARO;
}

Alt::Alt(double _t, const Vector1d &_z, const Matrix1d &_R)
{
    t = _t;
    // z(0) = _z;
    // R(0) = _R;
    z = _z;
    R = _R;
    type = ALT;
}

Gnss::Gnss(double _t, const Vector6d& _z, const Matrix6d& _R) :
    p(z.data()),
    v(z.data()+3)
{
    t = _t;
    type = GNSS;
    z = _z;
    R = _R;
}

Mocap::Mocap(double _t, const xform::Xformd &_z, const Matrix6d &_R) :
    z(_z),
    R(_R)
{
    t = _t;
    type = MOCAP;
}

Aruco::Aruco(double _t, const Eigen::Vector3d &_z, const Eigen::Matrix3d &_R,
             const quat::Quatd &_q_c2a, const Matrix1d& _yaw_R) :
    z(_z),
    R(_R),
    q_c2a(_q_c2a),
    yaw_R(_yaw_R)
{
    t = _t;
    type = ARUCO;
}

ZeroVel::ZeroVel(double _t)
{
    t = _t;
    type = ZERO_VEL;
}
}
