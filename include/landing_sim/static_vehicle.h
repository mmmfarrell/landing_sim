#pragma once

#include <Eigen/Core>

#include "multirotor_sim/state.h"
#include "multirotor_sim/vehicle_base.h"

namespace  multirotor_sim
{

class StaticVehicle : public VehicleBase
{
public:
  StaticVehicle() {}

  void step(const double& dt) {}
  void landmarkLocations(std::vector<Vector3d>& pts)
  {
    pts.clear();
    pts.push_back(Vector3d(2., 0., 0.));
    pts.push_back(Vector3d(1., 0., 0.));
    pts.push_back(Vector3d(0., 0., 0.));
    pts.push_back(Vector3d(0., 2., 0.));
  }
};

}
