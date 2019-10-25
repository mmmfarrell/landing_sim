#include <iostream>
#include <string>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "geometry/xform.h"

#include "ekf/logger.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/utils.h"
// #include "ekf/yaml.h"

#include "ekf/ekf_sim.h"
#include "landing_sim/unicycle_vehicle.h"

using namespace Eigen;
using namespace xform;
using namespace multirotor_sim;

int main()
{
  // Setup sim
  std::string sim_params_yaml_file = "../params/landing_est/sim_params.yaml";
  bool show_progress_bar = true;
  Simulator sim(show_progress_bar);
  sim.load(sim_params_yaml_file);
  Logger log("/tmp/landing_sim.bin");

  std::string est_param_file = "../params/ekf/ekf.yaml";
  EKF_SIM estimator;
  estimator.init(est_param_file);
  sim.register_estimator(&estimator);

  std::string veh_param_file = "../params/unicycle_params.yaml";
  UnicycleVehicle veh(veh_param_file);
  sim.use_custom_vehicle(&veh);

  // Run sim until done and log data
  while (sim.run())
  {
    // time
    log.log(sim.t_);

    // Uav States
    log.logVectors(sim.state().p, sim.state().q.arr_, sim.state().q.euler(), sim.state().v);
    log.logVectors(sim.accel_bias_, sim.gyro_bias_);


    ekf::State est_state = estimator.getEstimate();
    log.logVectors(est_state.p, est_state.q.arr_, est_state.q.euler(), est_state.v);
    log.logVectors(est_state.ba, est_state.bg);
    log.logVectors(est_state.gp, est_state.gv);
    log.log(est_state.gatt);
    log.log(est_state.gw);

    ekf::dxVec est_P = estimator.getCovariance();
    log.logVectors(est_P);
  }
  return 0;
}

