#include <iostream>
#include <string>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "geometry/xform.h"

#include "logger.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/utils.h"

#include "estimator.h"

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

  Estimator estimator;
  sim.register_estimator(&estimator);

  // Reference Trajectory
  //double x_width, y_width, z_width, period, throttle_eq;
  //get_yaml_node("x_width", sim_params_yaml_file, x_width);
  //get_yaml_node("y_width", sim_params_yaml_file, y_width);
  //get_yaml_node("z_width", sim_params_yaml_file, z_width);
  //get_yaml_node("period", sim_params_yaml_file, period);
  //get_yaml_node("throttle_eq", sim_params_yaml_file, throttle_eq);

  //Figure8 figure8(x_width, y_width, z_width, period, sim.state(),
                  //throttle_eq);
  //sim.use_custom_trajectory(&figure8);

  //// LQR Controller
  //std::string lqr_params_yaml_file = "../params/lqr/lqr_params_figure8.yaml";
  //LQRController lqr_controller(lqr_params_yaml_file);
  //sim.use_custom_controller(&lqr_controller);

  // Initialize uav to first commanded state
//  sim.run();
//  sim.state().arr = sim.commanded_state().arr;

  // Run sim until done and log data
  while (sim.run())
  {
    log.log(sim.t_);
    log.logVectors(sim.state().arr, sim.input(), sim.commanded_state().arr,
                   sim.reference_input());
  }
  return 0;
}

