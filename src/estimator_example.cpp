#include <iostream>
#include <string>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "geometry/xform.h"

#include "logger.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/utils.h"

#include "estimator.h"
#include "landing_sim/unicycle_vehicle.h"

using namespace Eigen;
using namespace xform;
using namespace multirotor_sim;

int main()
{
  // Setup sim
  std::string sim_params_yaml_file = "../params/basic_sim_params.yaml";
  bool show_progress_bar = true;
  Simulator sim(show_progress_bar);
  sim.load(sim_params_yaml_file);
  Logger log("/tmp/landing_sim.bin");

  std::string est_param_file = "../params/estimator_params.yaml";
  Estimator estimator(est_param_file);
  sim.register_estimator(&estimator);

  std::string veh_param_file = "../params/unicycle_params.yaml";
  UnicycleVehicle veh(veh_param_file);
  sim.use_custom_vehicle(&veh);

  Eigen::Matrix<double, 19, 1> p_diag;
  // Run sim until done and log data
  while (sim.run())
  {
    // time
    log.log(sim.t_);

    // Uav States
    log.logVectors(sim.state().arr, sim.input(), sim.commanded_state().arr,
                   sim.reference_input());

    // Landing vehicle states
    // TODO dont need to log the landmarks each step
    log.logVectors(veh.x_, veh.landmarks_body_);

    // Estimator states
    p_diag = estimator.P_.diagonal();
    log.logVectors(estimator.xhat_, p_diag);
  }
  return 0;
}

