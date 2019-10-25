#include "ekf/ekf.h"

#define T transpose()

using namespace Eigen;
namespace ekf
{


const dxMat EKF::I_BIG = dxMat::Identity();

EKF::EKF() :
  xbuf_(100)
{

}

EKF::~EKF()
{
  for (int i = 0; i < NUM_LOGS; i++)
    delete logs_[i];
}

void EKF::load(const std::string &filename)
{
  // Constant Parameters
  get_yaml_node("max_landmarks", filename, max_landmarks_);
  if (max_landmarks_ > ErrorState::MAX_LMS)
    max_landmarks_ = ErrorState::MAX_LMS;
  else if (max_landmarks_ < 0)
    max_landmarks_ = 0;

  get_yaml_eigen("p_b2g", filename, p_b2g_);

  // Camera Parameters
  get_yaml_eigen("cam_K", filename, cam_K_);
  cam_K_inv_ = cam_K_.inverse();
  get_yaml_eigen("p_b2c", filename, p_b2c_);
  get_yaml_eigen("q_b2c", filename, q_b2c_.arr_);
  q_b2c_.normalize();

  Qx_.setZero();
  Eigen::Matrix<double, ErrorState::DLMS, ErrorState::DLMS> Qx_states;
  get_yaml_diag("Qx", filename, Qx_states);
  Qx_.topLeftCorner(ErrorState::DLMS, ErrorState::DLMS) = Qx_states;
  get_yaml_eigen("Qx_lms", filename, Qx_lms_);

  P().setZero();
  Eigen::Matrix<double, ErrorState::DLMS, ErrorState::DLMS> P0_states;
  get_yaml_diag("P0", filename, P0_states);
  P().topLeftCorner(ErrorState::DLMS, ErrorState::DLMS) = P0_states;
  P0_yaw_ = P()(ErrorState::DQ + 2, ErrorState::DQ + 2);
  get_yaml_eigen("P0_lms", filename, P0_lms_);

  get_yaml_diag("R_zero_vel", filename, R_zero_vel_);
  get_yaml_diag("R_pixel_lms", filename, R_lms_);

  // Partial Update
  lambda_vec_.setZero();
  Eigen::Matrix<double, ErrorState::DLMS, 1> lambda_states;
  get_yaml_eigen("lambda", filename, lambda_states);
  lambda_vec_.head(ErrorState::DLMS) = lambda_states;
  get_yaml_eigen("lambda_lms", filename, lambda_lms_);
  for (unsigned int i = 0; i < max_landmarks_; ++i)
  {
    unsigned int lm_idx = ErrorState::DLMS + 3 * i; 
    lambda_vec_.segment<3>(lm_idx) = lambda_lms_;
  }

  const dxVec ones = dxVec::Constant(1.0);
  lambda_mat_ = ones * lambda_vec_.transpose() + lambda_vec_ * ones.transpose() -
                lambda_vec_ * lambda_vec_.transpose();

  // Measurement Flags
  get_yaml_node("enable_partial_update", filename, enable_partial_update_);
  get_yaml_node("enable_out_of_order", filename, enable_out_of_order_);
  get_yaml_node("use_mocap", filename, use_mocap_);
  get_yaml_node("use_gnss", filename, use_gnss_);
  get_yaml_node("use_baro", filename, use_baro_);
  get_yaml_node("use_range", filename, use_range_);
  get_yaml_node("use_zero_vel", filename, use_zero_vel_);
  get_yaml_node("use_aruco", filename, use_aruco_);
  get_yaml_node("use_lms", filename, use_lms_);

  // Armed Check
  get_yaml_node("enable_arm_check", filename, enable_arm_check_);
  get_yaml_node("is_flying_threshold", filename, is_flying_threshold_);

  // load initial state
  double ref_heading;
  get_yaml_node("ref_heading", filename, ref_heading);
  q_n2I_ = quat::Quatd::from_euler(0, 0, M_PI/180.0 * ref_heading);

  ref_lla_set_ = false;
  bool manual_ref_lla;
  get_yaml_node("manual_ref_lla", filename, manual_ref_lla);
  if (manual_ref_lla)
  {
    Vector3d ref_lla;
    get_yaml_eigen("ref_lla", filename, ref_lla);
    std::cout << "Set ref lla: " << ref_lla.transpose() << std::endl;
    ref_lla.head<2>() *= M_PI/180.0; // convert to rad
    xform::Xformd x_e2n = x_ecef2ned(lla2ecef(ref_lla));
    x_e2I_.t() = x_e2n.t();
    x_e2I_.q() = x_e2n.q() * q_n2I_;

    // initialize the estimated ref altitude state
    x().ref = ref_lla(2);
    ref_lat_radians_ = ref_lla(0);
    ref_lon_radians_ = ref_lla(1);

    ref_lla_set_ = true;
  }

  ground_pressure_ = 0.;
  ground_temperature_ = 0.;
  update_baro_ = false;
  get_yaml_node("update_baro_velocity_threshold", filename, update_baro_vel_thresh_);

  get_yaml_eigen("x0", filename, x0_.arr());

  goal_initialized_ = false;

  initLog(filename);
}

void EKF::initLog(const std::string &filename)
{
  get_yaml_node("enable_log", filename, enable_log_);
  get_yaml_node("log_prefix", filename, log_prefix_);

  std::experimental::filesystem::create_directories(log_prefix_);

  logs_.resize(NUM_LOGS);
  for (int i = 0; i < NUM_LOGS; i++)
    logs_[i] = new Logger(log_prefix_ + "/" + log_names_[i] + ".bin");
}

void EKF::initialize(double t)
{
  x().t = t;
  x().x = x0_;
  x().v.setZero();
  x().ba.setZero();
  x().bg.setZero();
  x().bb = 0.;
  if (ref_lla_set_)
    x().ref = x().ref;
  else
    x().ref = 0.;
  x().gp.setZero();
  x().gv.setZero();
  x().gatt = 0.;
  x().gw = 0.;
  x().lms.setZero();
  x().a = -gravity;
  x().w.setZero();
  is_flying_ = false;
  armed_ = false;
}

void EKF::propagate(const double &t, const Vector6d &imu, const Matrix6d &R)
{
  if (std::isnan(x().t))
  {
    initialize(t);
    return;
  }

  double dt = t - x().t;
  assert(dt >= 0);
  if (dt < 1e-6)
    return;

  dynamics(x(), imu, dx_, true);

  // do the state propagation
  xbuf_.next().x = x() + dx_ * dt;
  xbuf_.next().x.t = t;
  xbuf_.next().x.imu = imu;

  // Only do matrix math on the active states
  int num_states;
  if (goal_initialized_)
  {
    const int num_landmarks = landmark_ids_.size();
    num_states = ErrorState::DLMS + 3 * num_landmarks;
  }
  else
  {
    num_states = ErrorState::DGP;
  }

  auto Asmall = A_.topLeftCorner(num_states, num_states);
  auto Bsmall = B_.topRows(num_states);
  auto Ismall = I_BIG.topLeftCorner(num_states, num_states);
  auto Psmall = P().topLeftCorner(num_states, num_states);

  // discretize jacobians (first order)
  Asmall = Ismall + Asmall*dt;
  Bsmall = Bsmall*dt;

  CHECK_NAN(P());
  CHECK_NAN(A_);
  CHECK_NAN(B_);
  CHECK_NAN(Qx_);

  xbuf_.next().P = P();
  xbuf_.next().P.topLeftCorner(num_states, num_states) =
    Asmall*Psmall*Asmall.T + Bsmall*R*Bsmall.T + Qx_*dt*dt; // covariance propagation
  CHECK_NAN(xbuf_.next().P);

  // // discretize jacobians (first order)
  // A_ = I_BIG + A_*dt;
  // B_ = B_*dt;

  // CHECK_NAN(P());
  // CHECK_NAN(A_);
  // CHECK_NAN(B_);
  // CHECK_NAN(Qx_);

  // xbuf_.next().P =
    // A_*P()*A_.T + B_*R*B_.T + Qx_; // covariance propagation
    // // A_*P()*A_.T + B_*R*B_.T + Qx_*dt*dt; // covariance propagation
  // CHECK_NAN(xbuf_.next().P);

  xbuf_.advance();
  Qu_ = R; // copy because we might need it later.

  wrapAngle(x().gatt);

  if (enable_log_)
  {
    logs_[LOG_STATE]->logVectors(x().arr, x().q.euler());
    logs_[LOG_COV]->log(x().t);
    logs_[LOG_COV]->logVectors(P());
    logs_[LOG_IMU]->log(t);
    logs_[LOG_IMU]->logVectors(imu);
  }
}

void EKF::run()
{
  meas::MeasSet::iterator nmit = getOldestNewMeas();
  if (nmit == meas_.end())
    return;

  // rewind state history
  if (!xbuf_.rewind((*nmit)->t))
    throw std::runtime_error("unable to rewind enough, expand STATE_BUF");

  // re-propagate forward, integrating measurements on the way
  while (nmit != meas_.end())
  {
    update(*nmit);
    nmit++;
  }

  // clear off measurements older than the state history
  while (xbuf_.begin().x.t > (*meas_.begin())->t)
    meas_.erase(meas_.begin());

  logState();
}

void EKF::update(const meas::Base* m)
{
  if (m->type == meas::Base::IMU)
  {
    const meas::Imu* z = dynamic_cast<const meas::Imu*>(m);
    propagate(z->t, z->z, z->R);

    if (!is_flying_)
      zeroVelUpdate(z->t);
  }
  else if (!std::isnan(x().t))
  {
    propagate(m->t, x().imu, Qu_);
    switch(m->type)
    {
    case meas::Base::GNSS:
      {
        const meas::Gnss* z = dynamic_cast<const meas::Gnss*>(m);
        gnssUpdate(*z);
        break;
      }
    case meas::Base::MOCAP:
      {
        const meas::Mocap* z = dynamic_cast<const meas::Mocap*>(m);
        mocapUpdate(*z);
        break;
      }
    default:
      break;
    }
  }
  cleanUpMeasurementBuffers();
}

meas::MeasSet::iterator EKF::getOldestNewMeas()
{
  meas::MeasSet::iterator it = meas_.begin();
  while (it != meas_.end() && (*it)->handled)
  {
    it++;
  }
  return it;
}

bool EKF::measUpdate(const VectorXd &res, const MatrixXd &R, const MatrixXd &H)
{
  int num_states;
  if (goal_initialized_)
  {
    const int num_landmarks = landmark_ids_.size();
    num_states = ErrorState::DLMS + 3 * num_landmarks;
  }
  else
  {
    num_states = ErrorState::DGP;
  }

  int size = res.rows();
  auto K = K_.leftCols(size);

  // Create small versions of each matrix to only do math on valid states
  // Note this is not necessary. Estimator works great without this, it just
  // speeds it up a bunch when there are consistently less landmarks tracked
  // than MAXLANDMARKS
  auto Ksmall = K.topRows(num_states);
  auto Hsmall = H.leftCols(num_states);
  auto Psmall = P().topLeftCorner(num_states, num_states);
  // auto xsmall = x().topRows(num_states);
  // auto lam_vec_small = lambda_vec_.topRows(num_states);
  auto lam_mat_small = lambda_mat_.topLeftCorner(num_states, num_states);
  auto Ismall = I_BIG.topLeftCorner(num_states, num_states);


  ///TODO: perform covariance gating
  // MatrixXd innov = (H*P()*H.T + R).inverse();
  MatrixXd innov = (Hsmall*Psmall*Hsmall.T + R).inverse();

  CHECK_NAN(H); CHECK_NAN(R); CHECK_NAN(P());
  // K = P() * H.T * innov;
  Ksmall = Psmall * Hsmall.T * innov;
  CHECK_NAN(K);

  if (enable_partial_update_)
  {
    // Apply Fixed Gain Partial update per
    // "Partial-Update Schmidt-Kalman Filter" by Brink
    // Modified to operate inline and on the manifold

    // Here we do not operate on a small version of x because only the full
    // version has been overloaded for a boxplus
    x() += lambda_vec_.asDiagonal() * K * res;
    // dxMat ImKH = I_BIG - K*H;
    // P() += lambda_mat_.cwiseProduct(ImKH*P()*ImKH.T + K*R*K.T - P());

    // xsmall += lam_vec_small.asDiagonal() * K * res;
    auto ImKH = Ismall - Ksmall*Hsmall;
    Psmall += lam_mat_small.cwiseProduct(ImKH*Psmall*ImKH.T + Ksmall*R*Ksmall.T - Psmall);
  }
  else
  {
    x() += K * res;
    // dxMat ImKH = I_BIG - K*H;
    // P() = ImKH*P()*ImKH.T + K*R*K.T;
    auto ImKH = Ismall - Ksmall*Hsmall;
    Psmall += ImKH*Psmall*ImKH.T + Ksmall*R*Ksmall.T;
  }

  wrapAngle(x().gatt);
  CHECK_NAN(P());
  return true;
}

void EKF::imuCallback(const double &t, const Vector6d &z, const Matrix6d &R)
{
  if (!is_flying_)
    checkIsFlying();

  ///TODO: make thread-safe (wrap in mutex)
  if (enable_out_of_order_)
  {
    imu_meas_buf_.push_back(meas::Imu(t, z, R));
    meas_.insert(meas_.end(), &imu_meas_buf_.back());
    run(); // For now, run on the IMU heartbeat (could be made multi-threaded)
  }
  else
  {
    propagate(t, z, R);
    if (!is_flying_)
      zeroVelUpdate(t);
  }

  if (enable_log_)
  {
    logs_[LOG_IMU]->log(t);
    logs_[LOG_IMU]->logVectors(z);
  }

}

void EKF::baroCallback(const double &t, const double &z, const double &R,
                       const double &temp)
{
  if (enable_out_of_order_)
  {
    std::cout << "ERROR OUT OF ORDER BARO NOT IMPLEMENTED" << std::endl;
  }
  else
    baroUpdate(meas::Baro(t, z, R, temp));
}

void EKF::rangeCallback(const double& t, const double& z, const double& R)
{
  if (enable_out_of_order_)
  {
    std::cout << "ERROR OUT OF ORDER RANGE NOT IMPLEMENTED" << std::endl;
  }
  else
    rangeUpdate(meas::Range(t, z, R));
}

void EKF::gnssCallback(const double &t, const Vector6d &z, const Matrix6d &R)
{
  if (!ref_lla_set_)
    return;

  if (enable_out_of_order_)
  {
    gnss_meas_buf_.push_back(meas::Gnss(t, z, R));
    meas_.insert(meas_.end(), &gnss_meas_buf_.back());
  }
  else
    gnssUpdate(meas::Gnss(t, z, R));

  if (enable_log_)
  {
    logs_[LOG_LLA]->log(t);
    logs_[LOG_LLA]->logVectors(ecef2lla((x_e2I_ * x().x).t()));
    logs_[LOG_LLA]->logVectors(ecef2lla(z.head<3>()));
  }
}

void EKF::mocapCallback(const double& t, const xform::Xformd& z, const Matrix6d& R)
{
  if (enable_out_of_order_)
  {
    mocap_meas_buf_.push_back(meas::Mocap(t, z, R));
    meas_.insert(meas_.end(), &mocap_meas_buf_.back());
  }
  else
    mocapUpdate(meas::Mocap(t, z, R));


  if (enable_log_)
  {
    logs_[LOG_REF]->log(t);
    logs_[LOG_REF]->logVectors(z.arr(), z.q().euler());
  }
}

void EKF::arucoCallback(const double& t, const Eigen::Vector3d& z,
                        const Eigen::Matrix3d& R, const quat::Quatd& q_c2a,
                        const Matrix1d& yaw_R)
{
  if (enable_out_of_order_)
  {
    std::cout << "ERROR OUT OF ORDER ARUCO NOT IMPLEMENTED" << std::endl;
  }
  else
    arucoUpdate(meas::Aruco(t, z, R, q_c2a, yaw_R));
}

void EKF::landmarksCallback(const double& t, const ImageFeat& z)
{
  if (!goal_initialized_)
  {
    return;
  }

  std::list<int>::iterator it = landmark_ids_.begin();

  // std::cout << "EKF LM Callback start ids: " << std::endl;
  // while (it != landmark_ids_.end())
  // {
    // std::cout << *it << ", ";
    // it++;
  // }
  // std::cout << std::endl;
  // it = landmark_ids_.begin();
  // std::cout << "LMS: " << x().lms << std::endl;

  int idx = 0;
  int num_landmarks = z.pixs.size();
  if (num_landmarks > max_landmarks_)
  {
    num_landmarks = max_landmarks_;
  }

  while (idx < num_landmarks)
  {
    const int lm_id = z.feat_ids[idx];
    const Eigen::Vector2d lm_pix = z.pixs[idx];
    if (it == landmark_ids_.end())
    {
      initLandmark(lm_id, lm_pix);
      idx++;
    }
    else
    {
      const std::list<int>::iterator prev_lm_it = it;
      const int expected_lm_id = *prev_lm_it;
      it++;
      if (expected_lm_id != lm_id)
      {
        removeLandmark(idx, prev_lm_it);
      }
      else
      {
        landmarkUpdate(idx, lm_pix);
        idx++;
      }
    }
  }
}

void EKF::printLmIDs()
{
  std::list<int>::iterator it;

  std::cout << "List: ";
  for (it = landmark_ids_.begin(); it != landmark_ids_.end(); it++)
  {
    std::cout << *it << ", ";
  }
  std::cout << std::endl;
}

void EKF::printLmXhat()
{
  PRINTMAT(x().lms);
  // std::cout << "lm xhat: " << std::endl
            // << xhat_.bottomRows(MAXLANDMARKS * 3) << std::endl;
}

void EKF::printLmPhat()
{
  std::cout << "lm Phat: " << std::endl
            << P().bottomRightCorner(ErrorState::MAX_LMS * 3, ErrorState::MAX_LMS * 3)
            << std::endl;
}

void EKF::initLandmark(const int& id, const Vector2d& pix)
{
  // lm_idx is 0 indexed
  const int lm_idx = landmark_ids_.size();

  std::cout << "init idx: " << lm_idx << std::endl;

  // add the id to the list of ids tracked
  landmark_ids_.push_back(id);

  // initialize estimator xhat and phat
  const int LM_IDX = ErrorState::DLMS + 3 * lm_idx;
  P().block<3, 3>(LM_IDX, LM_IDX) = P0_lms_.asDiagonal();
  Qx_.block<3, 3>(LM_IDX, LM_IDX) = Qx_lms_.asDiagonal();

  const Eigen::Matrix3d R_b2c = q_b2c_.R();
  const Eigen::Matrix3d R_I2b = x().q.R();

  Eigen::Vector3d pix_homo(pix(0), pix(1), 1.);
  Eigen::Vector3d unit_vec_veh_frame =
      R_I2b.transpose() * R_b2c.transpose() * cam_K_inv_ * pix_homo;

  const Eigen::Vector3d p_c_b_I = R_I2b * p_b2c_;
  // const double expected_altitude = xhat_(xGOAL_POS + 2) - p_c_b_I(2);
  const double expected_altitude = -x().gp(2) - x().p(2) - p_c_b_I(2);

  Eigen::Vector3d scaled_vec_veh_frame =
      (expected_altitude / unit_vec_veh_frame(2)) * unit_vec_veh_frame;
  Eigen::Vector3d p_i_c_v = scaled_vec_veh_frame;
  const Eigen::Vector3d p_i_v_v = p_i_c_v + p_c_b_I;

  const double theta_g = x().gatt;
  Eigen::Matrix3d R_v2g = Eigen::Matrix3d::Identity();
  R_v2g.topLeftCorner(2, 2) = rotm2dItoB(theta_g);

  // Eigen::Vector3d p_g_v_v = xhat_.segment<3>(xGOAL_POS);
  Eigen::Vector3d p_g_v_v(x().gp(0), x().gp(1), -x().gp(2)-x().p(2));
  Eigen::Vector3d p_i_g_g = R_v2g * (p_i_v_v - p_g_v_v);

  // Init state with estimate
  // xhat_.block<3, 1>(xLM_IDX, 0) = p_i_g_g;
  x().lms.block<3, 1>(0, lm_idx) = p_i_g_g.transpose();
  x().lms(2, lm_idx) = 0.;

  std::cout << "lms: " << x().lms << std::endl;
}

void EKF::removeLandmark(const int& lm_idx, const std::list<int>::iterator it)
{
  std::cout << "Remove lm idx: " << lm_idx << std::endl;
  std::cout << "lms before: " << x().lms << std::endl;
  landmark_ids_.erase(it);

  // Move up the bottom rows to cover up the values corresponding to the removed
  // lm
  using E = ErrorState;
  const int LM_IDX = E::DLMS + 3 * lm_idx;
  const int num_rows = E::SIZE - LM_IDX - 3;
  const int num_cols = num_rows;

  // const int num_lms_right = max_landmarks_ - lm_idx - 1;
  const int num_lms_right = E::MAX_LMS - lm_idx - 1;

  // xhat_.block(LM_IDX, 0, num_rows, 1) = xhat_.bottomRows(num_rows);
  // // Zero out the unintialized xhat terms (NOT necessary)
  // xhat_.bottomRows(3).setZero();

  x().lms.block(0, lm_idx, 3, num_lms_right) = x().lms.rightCols(num_lms_right);
  // Zero out the unintialized xhat terms (NOT necessary)
  x().lms.rightCols(1).setZero();

  // Move covariance up and then left to preserve cross terms
  P().block(LM_IDX, 0, num_rows, E::SIZE) = P().bottomRightCorner(num_rows, E::SIZE);
  P().block(0, LM_IDX, E::SIZE, num_cols) = P().bottomRightCorner(E::SIZE, num_cols);
  // Zero out the unintialized covariance terms (necessary)
  P().bottomRightCorner(3, E::SIZE).setZero();
  P().bottomRightCorner(E::SIZE, 3).setZero();

  Qx_.block(LM_IDX, 0, num_rows, E::SIZE) = Qx_.bottomRightCorner(num_rows, E::SIZE);
  Qx_.block(0, LM_IDX, E::SIZE, num_cols) = Qx_.bottomRightCorner(E::SIZE, num_cols);
  // Zero out the unintialized covariance terms (necessary)
  Qx_.bottomRightCorner(3, E::SIZE).setZero();
  Qx_.bottomRightCorner(E::SIZE, 3).setZero();

  std::cout << "lms after: " << x().lms << std::endl;
}

void EKF::landmarkUpdate(const int& idx, const Vector2d& pix)
{
  // Landmarks are 0 indexed
  const int LM_IDX = ErrorState::DLMS + 3 * idx;

  // Camera Params
  const double fx = cam_K_(0, 0);
  const double fy = cam_K_(1, 1);
  const double cx = cam_K_(0, 2);
  const double cy = cam_K_(1, 2);

  // Constants
  static const Eigen::Vector3d e1(1., 0., 0.);
  static const Eigen::Vector3d e2(0., 1., 0.);
  static const Eigen::Vector3d e3(0., 0., 1.);

  const Eigen::Matrix3d R_b2c = q_b2c_.R();
  const Eigen::Matrix3d R_I2b = x().q.R();

  const double theta_g = x().gatt;
  const Eigen::Matrix2d R_I2g_2d = rotm2dItoB(theta_g);
  Eigen::Matrix3d R_I2g = Eigen::Matrix3d::Identity();
  R_I2g.topLeftCorner(2, 2) = R_I2g_2d;

  // const Eigen::Vector3d p_i_g_g = x.segment<3>(xLM_IDX);
  const Eigen::Vector3d p_i_g_g = x().lms.block<3, 1>(0, idx);
  const Eigen::Vector3d p_i_g_v = R_I2g.transpose() * p_i_g_g;

  // const Eigen::Vector3d p_g_v_v = x.segment<3>(Estimator::xGOAL_POS);
  Eigen::Vector3d p_g_v_v(x().gp(0), x().gp(1), -x().gp(2)-x().p(2));
  const Eigen::Vector3d p_i_v_v = p_i_g_v + p_g_v_v;

  const Eigen::Vector3d p_i_c_c = R_b2c * (R_I2b * p_i_v_v - p_b2c_);

  // Measurement Model
  const double px_hat = fx * (p_i_c_c(0) / p_i_c_c(2)) + cx;
  const double py_hat = fy * (p_i_c_c(1) / p_i_c_c(2)) + cy;
  const Vector2d zhat(px_hat, py_hat);
  const Vector2d r = pix - zhat;
  // std::cout << "update lm id: " << idx << std::endl;
  // PRINTMAT(pix);
  // PRINTMAT(zhat);

  using E = ErrorState;

  Eigen::Matrix<double, 2, E::NDX> H;
  H.setZero();

  // Measurement Model Jacobian
  // const Eigen::Matrix3d d_R_d_phi = dRIBdPhi(phi, theta, psi);
  // const Eigen::Matrix3d d_R_d_theta = dRIBdTheta(phi, theta, psi);
  // const Eigen::Matrix3d d_R_d_psi = dRIBdPsi(phi, theta, psi);

  // const Eigen::Vector3d RdRdPhip = R_b2c * d_R_d_phi * p_i_v_v;
  // const double dpx_dphi =
      // (fx * RdRdPhip(0) / p_i_c_c(2)) -
      // (fx * RdRdPhip(2) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  // const Eigen::Vector3d RdRdThetap = R_b2c * d_R_d_theta * p_i_v_v;
  // const double dpx_dtheta =
      // (fx * RdRdThetap(0) / p_i_c_c(2)) -
      // (fx * RdRdThetap(2) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  // const Eigen::Vector3d RdRdPsip = R_b2c * d_R_d_psi * p_i_v_v;
  // const double dpx_dpsi =
      // (fx * RdRdPsip(0) / p_i_c_c(2)) -
      // (fx * RdRdPsip(2) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  const Eigen::Matrix3d blah = -R_b2c * R_I2b * skew(p_i_v_v);
  const Eigen::Vector3d blah1 = e1.transpose() * blah;
  const Eigen::Vector3d blah3 = e3.transpose() * blah;
  const Eigen::Vector3d dpx_dq =
      (fx * blah1 / p_i_c_c(2)) -
      (fx * blah3 * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));

  const Eigen::Vector3d dpx_dp =
      ((fx * e1.transpose() * R_b2c * R_I2b) / p_i_c_c(2)) -
      ((fx * e3.transpose() * R_b2c * R_I2b * p_i_c_c(0)) /
       (p_i_c_c(2) * p_i_c_c(2)));

  H.setZero();
  H.block<1, 3>(0, E::DQ) = dpx_dq;
  H.block<1, 2>(0, E::DGP) = dpx_dp.head(2);
  H(0, E::DGP + 2) = -dpx_dp(2);
  H(0, E::DP + 2) = -dpx_dp(2);

  // const double dpy_dphi =
      // (fy * RdRdPhip(1) / p_i_c_c(2)) -
      // (fy * RdRdPhip(2) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  // const double dpy_dtheta =
      // (fy * RdRdThetap(1) / p_i_c_c(2)) -
      // (fy * RdRdThetap(2) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  // const double dpy_dpsi =
      // (fy * RdRdPsip(1) / p_i_c_c(2)) -
      // (fy * RdRdPsip(2) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  const Eigen::Vector3d blah2 = e2.transpose() * blah;
  const Eigen::Vector3d dpy_dq =
      (fy * blah2 / p_i_c_c(2)) -
      (fy * blah3 * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));

  const Eigen::Vector3d dpy_dp =
      ((fy * e2.transpose() * R_b2c * R_I2b) / p_i_c_c(2)) -
      ((fy * e3.transpose() * R_b2c * R_I2b * p_i_c_c(1)) /
       (p_i_c_c(2) * p_i_c_c(2)));

  H.block<1, 3>(1, E::DQ) = dpy_dq;
  H.block<1, 2>(1, E::DGP) = dpy_dp.head(2);
  H(1, E::DGP + 2) = -dpy_dp(2);
  H(1, E::DP + 2) = -dpy_dp(2);

  // d / d theta_g
  const Eigen::Matrix2d d_R_d_theta_g_2d = dR2DdTheta(theta_g);
  // const Eigen::Matrix3d d_R_d_theta_g = dR3DdTheta(theta_g);
  Eigen::Matrix3d d_R_d_theta_g;
  d_R_d_theta_g.setZero();
  d_R_d_theta_g.topLeftCorner(2, 2) = d_R_d_theta_g_2d;
  const Vector3d d_theta_p_i_v_v = d_R_d_theta_g.transpose() * p_i_g_g;

  const Eigen::Vector3d RRdRdThetaP = R_b2c * R_I2b * d_theta_p_i_v_v;
  const double dpx_dtheta_g =
      (fx * RRdRdThetaP(0) / p_i_c_c(2)) -
      (fx * RRdRdThetaP(2) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  const double dpy_dtheta_g =
      (fy * RRdRdThetaP(1) / p_i_c_c(2)) -
      (fy * RRdRdThetaP(2) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  H(0, E::DGATT) = dpx_dtheta_g;
  H(1, E::DGATT) = dpy_dtheta_g;

  // d / d rxy
  const Eigen::Matrix3d d_r_p_i_v_v = R_I2g.transpose();

  const Eigen::Matrix3d RRdRdrp = R_b2c * R_I2b * d_r_p_i_v_v;
  const Eigen::Matrix<double, 1, 3> dpx_dr =
      (fx * RRdRdrp.block<1, 3>(0, 0) / p_i_c_c(2)) -
      (fx * RRdRdrp.block<1, 3>(2, 0) * p_i_c_c(0) / p_i_c_c(2) / p_i_c_c(2));
  const Eigen::Matrix<double, 1, 3> dpy_dr =
      (fy * RRdRdrp.block<1, 3>(1, 0) / p_i_c_c(2)) -
      (fy * RRdRdrp.block<1, 3>(2, 0) * p_i_c_c(1) / p_i_c_c(2) / p_i_c_c(2));
  H.block<1, 3>(0, LM_IDX) = dpx_dr;
  H.block<1, 3>(1, LM_IDX) = dpy_dr;

  /// TODO: Saturate r
  if (use_lms_)
    measUpdate(r, R_lms_, H);
}

void EKF::baroUpdate(const meas::Baro &z)
{
  if (!this->groundTempPressSet())
  {
    return;
  }
  else if (!update_baro_ || !is_flying_)
  {
    // Take the lowest pressure while I'm not flying as ground pressure
    // This has the effect of hopefully underestimating my altitude instead of
    // over estimating.
    if (z.z(0) < ground_pressure_)
    {
      ground_pressure_ = z.z(0);
      std::cout << "New ground pressure: " << ground_pressure_ << std::endl;
    }

    // check if we should start updating with the baro yet based on
    // velocity estimate
    if (x().v.norm() > update_baro_vel_thresh_)
      update_baro_ = true;

    return;
  }

  using Vector1d = Eigen::Matrix<double, 1, 1>;

  // // From "Small Unmanned Aircraft: Theory and Practice" eq 7.8
  const double g = 9.80665; // m/(s^2) gravity 
  const double R = 8.31432; // universal gas constant
  const double M = 0.0289644; // kg / mol. molar mass of Earth's air

  const double altitude = -x().p(2);
  const double baro_bias = x().bb;

  // From "Small Unmanned Aircraft: Theory and Practice" eq 7.9
  // const double rho = M * ground_pressure_ / R / ground_temperature_;
  const double rho = M * ground_pressure_ / R / z.temp;

  const double press_hat = ground_pressure_ - rho * g * altitude + baro_bias;

  const Vector1d zhat(press_hat);
  Vector1d r = z.z - zhat;

  typedef ErrorState E;

  Matrix<double, 1, E::NDX> H;
  H.setZero();
  H(0, E::DP + 2) = rho * g;
  H(0, E::DBB) = 1.;

  /// TODO: Saturate r
  if (use_baro_)
    measUpdate(r, z.R, H);

  if (enable_log_)
  {
    logs_[LOG_BARO_RES]->log(z.t);
    logs_[LOG_BARO_RES]->logVectors(r, z.z, zhat);
    logs_[LOG_BARO_RES]->log(z.temp);
  }

}

void EKF::rangeUpdate(const meas::Range &z)
{
  // Assume that the earth is flat and that the range sensor is rigidly attached
  // to the UAV, so distance is dependent on the attitude of the UAV.
  // TODO this assumes that the laser is positioned at 0,0,0 in the body frame
  // of the UAV
  // TODO this also only updates if the UAV is pretty close to level and the
  // measurement model jacobian assumes that the measurement is not dependent
  // on roll or pitch at all
  using Vector1d = Eigen::Matrix<double, 1, 1>;

  const double altitude = -x().p(2);
  const double roll = x().q.roll();
  const double pitch = x().q.pitch();

  // const double level_threshold = 2. * M_PI / 180.; // 1 degree

  // Only update if UAV is close to level
  // if ((abs(roll) > level_threshold) || (abs(pitch) > level_threshold))
    // return;

  const Vector1d zhat(altitude / cos(roll) / cos(pitch)); // TODO roll/ pitch of drone
  Vector1d r = z.z - zhat; // residual

  // std::cout << "Laser Update: " << std::endl;
  // std::cout << "Altitude meas: " << z.z(0) << std::endl;
  // std::cout << "Altitude est: " << altitude << std::endl;

  typedef ErrorState E;

  Matrix<double, 1, E::NDX> H;
  H.setZero();
  H(0, E::DP + 2) = -1.;

  // Vector1d r_saturated
  double r_sat = 0.1;
  if (abs(r(0)) > r_sat)
  {
    double r_sign = (r(0) > 0) - (r(0) < 0);
    r(0) = r_sat * r_sign;
  }

  // TODO: Saturate r
  if (use_range_)
    measUpdate(r, z.R, H);

  if (enable_log_)
  {
    logs_[LOG_RANGE_RES]->log(z.t);
    logs_[LOG_RANGE_RES]->logVectors(r, z.z, zhat);
  }

}

void EKF::gnssUpdate(const meas::Gnss &z)
{
  const Vector3d w = x().w - x().bg;
  const Vector3d gps_pos_I = x().p + x().q.rota(p_b2g_);
  const Vector3d gps_vel_b = x().v + w.cross(p_b2g_);
  const Vector3d gps_vel_I = x().q.rota(gps_vel_b);

  // Update ref_lla based on current estimate
  Vector3d ref_lla(ref_lat_radians_, ref_lon_radians_, x().ref);
  xform::Xformd x_e2n = x_ecef2ned(lla2ecef(ref_lla));
  x_e2I_.t() = x_e2n.t();
  x_e2I_.q() = x_e2n.q() * q_n2I_;

  Vector6d zhat;
  zhat << x_e2I_.transforma(gps_pos_I),
          x_e2I_.rota(gps_vel_I);
  const Vector6d r = z.z - zhat; // residual

  const Matrix3d R_I2e = x_e2I_.q().R().T;
  const Matrix3d R_b2I = x().q.R().T;
  const Matrix3d R_e2b = R_I2e * R_b2I;

  const double sin_lat = sin(ref_lat_radians_);
  const double cos_lat = cos(ref_lat_radians_);
  const double sin_lon = sin(ref_lon_radians_);
  const double cos_lon = cos(ref_lon_radians_);
  const Vector3d dpEdRefAlt(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);

  typedef ErrorState E;

  Matrix<double, 6, E::NDX> H;
  H.setZero();
  H.block<3,3>(0, E::DP) = R_I2e; // dpE/dpI
  H.block<3,3>(0, E::DQ) = -R_e2b * skew(p_b2g_);
  H.block<3, 1>(0, E::DREF) = dpEdRefAlt;
  H.block<3,3>(3, E::DQ) = -R_e2b * skew(gps_vel_b); // dvE/dQI
  H.block<3,3>(3, E::DV) = R_e2b;
  H.block<3,3>(3, E::DBG) = R_e2b * skew(p_b2g_);

  /// TODO: Saturate r
  if (use_gnss_)
    measUpdate(r, z.R, H);

  if (enable_log_)
  {
    logs_[LOG_GNSS_RES]->log(z.t);
    logs_[LOG_GNSS_RES]->logVectors(r, z.z, zhat);
  }
}

void EKF::mocapUpdate(const meas::Mocap &z)
{
  xform::Xformd zhat = x().x;

  // TODO Do we need to fix "-" operator for Xformd?
  // Right now using piecewise subtraction
  // on position and attitude separately. This may be correct though because
  // our state is represented as R^3 x S^3 (position, quaterion) not SE3
  Vector6d r;
  r.segment<3>(0) = z.z.t_ - zhat.t_;
  r.segment<3>(3) = z.z.q_ - zhat.q_;

  typedef ErrorState E;
  Matrix<double, 6, E::NDX> H;
  H.setZero();
  H.block<3,3>(0, E::DP) = I_3x3;
  H.block<3,3>(3, E::DQ) = I_3x3;

  /// TODO: Saturate r
  if (use_mocap_)
  {
    measUpdate(r, z.R, H);
  }

  if (enable_log_)
  {
    logs_[LOG_MOCAP_RES]->log(z.t);
    logs_[LOG_MOCAP_RES]->logVectors(r, z.z.arr(), zhat.arr());
  }
}

void EKF::zeroVelUpdate(double t)
{
  // Update Zero velocity and zero altitude
  typedef ErrorState E;
  Matrix<double, 4, E::NDX> H;
  H.setZero();
  H.block<3,3>(0, E::DV) = I_3x3;
  H(3, E::DP + 2) = 1.;

  Vector4d r;
  r.head<3>() = -x().v;
  r(3) = -x().p(2);

  if (use_zero_vel_)
    measUpdate(r, R_zero_vel_, H);

  // Reset the uncertainty in yaw
  P().block<ErrorState::SIZE, 1>(0, ErrorState::DQ + 2).setZero();
  P().block<1, ErrorState::SIZE>(ErrorState::DQ + 2, 0).setZero();
  P()(ErrorState::DQ + 2, ErrorState::DQ + 2) = P0_yaw_;

  if (enable_log_)
  {
    logs_[LOG_ZERO_VEL_RES]->log(t);
    logs_[LOG_ZERO_VEL_RES]->logVectors(r);
  }
}

void EKF::arucoUpdate(const meas::Aruco &z)
{
  if (!goal_initialized_)
  {
    initGoal(z);
    std::cout << "Goal Initialized" << std::endl;
    return;
  }
  // TODO account for positional offset of camera

  // rotate goal position from inertial frame to body frame to camera frame
  const Vector2d goal_pos_2d = x().gp.head(2);
  const Vector3d goal_pos(goal_pos_2d(0), goal_pos_2d(1), -x().gp(2)-x().p(2));
  const Vector3d zhat = q_b2c_.rotp(x().q.rotp(goal_pos) - p_b2c_);
  const Vector3d r = z.z - zhat;

  const Matrix3d R_b2c = q_b2c_.R();
  const Matrix3d R_I2b = x().q.R();

  typedef ErrorState E;
  Matrix<double, 3, E::NDX> H;
  H.setZero();
  H.block<3, 1>(0, E::DP + 2) = -(R_b2c * R_I2b).rightCols(1);
  H.block<3, 2>(0, E::DGP) = (R_b2c * R_I2b).leftCols(2);
  H.block<3, 1>(0, E::DGP + 2) = -(R_b2c * R_I2b).rightCols(1);
  H.block<3, 3>(0, E::DQ) = R_b2c * R_I2b * skew(goal_pos);

  /// TODO: Saturate r
  if (use_aruco_)
    measUpdate(r, z.R, H);

  if (enable_log_)
  {
    logs_[LOG_ARUCO_RES]->log(z.t);
    logs_[LOG_ARUCO_RES]->logVectors(r, z.z, zhat);
  }

  // Update goal attitude (janky) TODO fixme
  // quat::Quatd q_a2g = quat::Quatd(M_PI, 0., M_PI / 2.);
  quat::Quatd q_a2g = quat::Quatd(0., 0., 0.);
  quat::Quatd q_I2g_meas = x().q * q_b2c_ * z.q_c2a * q_a2g;
  const double yaw_meas = q_I2g_meas.euler()(2);
  Vector1d r_yaw(yaw_meas - x().gatt);
  wrapAngle(r_yaw(0));

  Matrix<double, 1, E::NDX> H_yaw;
  H_yaw.setZero();
  H_yaw(0, E::DGATT) = 1.;

  /// TODO: Saturate r
  if (use_aruco_)
    measUpdate(r_yaw, z.yaw_R, H_yaw);
}

void EKF::initGoal(const meas::Aruco& z)
{
  // Get position of goal w.r.t vehicle frame, expressed in vehicle frame
  const Eigen::Matrix3d R_b2c = q_b2c_.R();
  const Eigen::Matrix3d R_I2b = x().q.R();
  const Vector3d p_g_c_c = z.z;
  const Vector3d p_g_v_v =
      R_I2b.transpose() * (R_b2c.transpose() * p_g_c_c + p_b2c_);
  x().gp.head(2) = p_g_v_v.head<2>();
  x().gp(2) = 0.;

  // Janky way to get ArUco yaw
  // quat::Quatd q_a2g = quat::Quatd(M_PI, 0., M_PI / 2.);
  quat::Quatd q_a2g = quat::Quatd(0., 0., 0.);
  quat::Quatd q_I2g_meas = x().q * q_b2c_ * z.q_c2a * q_a2g;
  const double yaw_meas = q_I2g_meas.euler()(2);
  x().gatt = yaw_meas;
  // x().gatt = 0.;

  // P should already be initialized
  goal_initialized_ = true;
}

void EKF::setRefLla(Vector3d ref_lla)
{
  if (ref_lla_set_)
    return;

  std::cout << "Set ref lla: " << ref_lla.transpose() << std::endl;
  ref_lla.head<2>() *= M_PI/180.0; // convert to rad
  xform::Xformd x_e2n = x_ecef2ned(lla2ecef(ref_lla));
  x_e2I_.t() = x_e2n.t();
  x_e2I_.q() = x_e2n.q() * q_n2I_;

  // initialize the estimated ref altitude state
  x().ref = ref_lla(2);
  ref_lat_radians_ = ref_lla(0);
  ref_lon_radians_ = ref_lla(1);

  ref_lla_set_ = true;

}

void EKF::wrapAngle(double& ang)
{
  while (ang > M_PI)
    ang -= 2 * M_PI;
  while (ang <= -M_PI)
    ang += 2 * M_PI;
}

void EKF::cleanUpMeasurementBuffers()
{
  // Remove all measurements older than our oldest state in the measurement buffer
  while ((*meas_.begin())->t < xbuf_.begin().x.t)
    meas_.erase(meas_.begin());
  while (imu_meas_buf_.front().t < xbuf_.begin().x.t)
    imu_meas_buf_.pop_front();
  while (mocap_meas_buf_.front().t < xbuf_.begin().x.t)
    mocap_meas_buf_.pop_front();
  while (gnss_meas_buf_.front().t < xbuf_.begin().x.t)
    gnss_meas_buf_.pop_front();
}

void EKF::setGroundTempPressure(const double& temp, const double& press)
{
  ground_temperature_ = temp;
  ground_pressure_ = press;
}

void EKF::checkIsFlying()
{
  bool okay_to_check = enable_arm_check_ ? armed_ : true;
  if (okay_to_check && x().a.norm() > is_flying_threshold_)
  {
    std::cout << "Now Flying!  Go Go Go!" << std::endl;
    is_flying_ = true;
  }
}


}
