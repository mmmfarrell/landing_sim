#include <iostream>
#include <string>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "geometry/xform.h"

#include "logger.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/utils.h"

#include "uav_estimator.h"
#include "landing_sim/unicycle_vehicle.h"

using namespace Eigen;
using namespace xform;
using namespace multirotor_sim;

int main()
{
  // Setup sim
  std::string sim_params_yaml_file = "../params/uav_est/sim_params.yaml";
  bool show_progress_bar = true;
  Simulator sim(show_progress_bar);
  sim.load(sim_params_yaml_file);
  Logger log("/tmp/landing_sim.bin");

  std::string est_param_file = "../params/uav_est/estimator_params.yaml";
  Estimator estimator(est_param_file);
  sim.register_estimator(&estimator);

  std::string veh_param_file = "../params/unicycle_params.yaml";
  UnicycleVehicle veh(veh_param_file);
  sim.use_custom_vehicle(&veh);

  Eigen::Matrix<double, Estimator::xZ, 1> p_diag;
  // Run sim until done and log data
  while (sim.run())
  {
    // time
    log.log(sim.t_);

    // Uav States
    //log.logVectors(sim.state().arr);
    log.logVectors(sim.state().p, sim.state().q.euler(), sim.state().v);

    // Estimator states
    p_diag = estimator.P_.diagonal();
    log.logVectors(estimator.xhat_, p_diag);
  }
  return 0;
}

