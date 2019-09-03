#include <iostream>
#include <string>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "geometry/xform.h"

#include "logger.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/utils.h"

#include "landing_estimator.h"
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

  std::string est_param_file = "../params/landing_est/estimator_params.yaml";
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
    Matrix1d mu_true(0.1);
    log.logVectors(sim.state().p, sim.state().q.euler(), sim.state().v, mu_true);

    //// Landing Goal states
    //// log x = [ px, py, vx, vy, theta, omega ]
    //log.logVectors(veh.x_);

    //// log lm positions (even though they never change)
    //// This is a 5 x 3 vector b/c (0, 0, 0) is in this too
    //log.logVectors(veh.landmarks_body_);

    // True states to log
    // p_{g/v}^{v}, rho, v_{g/I}^{g}, theta_{I}^{g}, omega_{g/I}^{g}, r_{i}^{g}, \rho_{i} 
    const Eigen::Vector2d p_g_v = veh.x_.head(2) - sim.state().p.head(2);
    Matrix1d rho;
    rho(0) = -1. / sim.state().p(2);
    const Eigen::Vector2d v_g_I = veh.x_.segment<2>(UnicycleVehicle::xVEL);
    const Eigen::Vector2d goal_theta_omega = veh.x_.segment<2>(UnicycleVehicle::xATT);
    Eigen::Matrix<double, Estimator::MAXLANDMARKS * 3, 1> landmarks;
    for (unsigned int i = 0; i < Estimator::MAXLANDMARKS; i++)
    {
      const int veh_lms_idx = i + 1;
      const int lms_vec_idx = 3 * i;
      //const int lms_rho_idx = 3 * i + 2;
      
      landmarks.block<3, 1>(lms_vec_idx, 0) = veh.landmarks_body_.block<1, 3>(veh_lms_idx, 0).transpose();
      //landmarks(lms_rho_idx) = 1. / (-sim.state().p(2) + veh.landmarks_body_(veh_lms_idx, 2));
    }

    log.logVectors(p_g_v, rho, v_g_I, goal_theta_omega, landmarks);

    //xPOS = 0,
    //xATT = 3,
    //xVEL = 6,
    //xMU = 9,
    //xGOAL_POS = 10,
    //xGOAL_RHO = 12,
    //xGOAL_VEL = 13,
    //xGOAL_ATT = 15,
    //xGOAL_OMEGA = 16,
    //xGOAL_LM = 17,
    //xZ = 29

    // Estimator states
    p_diag = estimator.P_.diagonal();
    log.logVectors(estimator.xhat_, p_diag);
  }
  return 0;
}

