#include <gtest/gtest.h>

#include "landing_estimator.h"
#include "measurement_models.h"

class RandomState : public ::testing::Test
{
public:
  RandomState()
  {
    xhat.setRandom();
  }
  Estimator::StateVec xhat;
  Estimator::MeasVec zhat;
  Estimator::MeasH H;
};

TEST_F(RandomState, goalPixelMeas_Dimension2)
{
  int meas_dims = 0;
  goalPixelMeasModel(xhat, meas_dims, zhat, H);

  const int true_meas_dims = 2;
  EXPECT_EQ(meas_dims, true_meas_dims);
}

TEST_F(RandomState, goalPixelMeas_NonZero)
{
  int meas_dims = 0;
  goalPixelMeasModel(xhat, meas_dims, zhat, H);
  EXPECT_TRUE((zhat.head(meas_dims).array() != 0.).all());
}

TEST_F(RandomState, goalPixelJac_ZerosWhereExpected)
{
  int meas_dims = 0;
  goalPixelMeasModel(xhat, meas_dims, zhat, H);

  // Goal meas model doesnt depend on UAV POS, VEL, DRAG or on GOAL ATT, OMEGA,
  // VEL, or LMS
  auto dHdPos = H.block<2, 3>(0, Estimator::xPOS);
  EXPECT_TRUE(dHdPos.isZero());

  auto dHdVel = H.block<2, 3>(0, Estimator::xVEL);
  EXPECT_TRUE(dHdVel.isZero());

  auto dHdDrag = H.block<2, 1>(0, Estimator::xMU);
  EXPECT_TRUE(dHdDrag.isZero());

  auto dHdGoal =
      H.block<2, Estimator::xZ - Estimator::xGOAL_VEL>(0, Estimator::xGOAL_VEL);
  EXPECT_TRUE(dHdGoal.isZero());

  // Goal meas model does depend on UAV ATT, GOAL POS & RHO
  // All entries should be different from 0
  auto dHdAtt = H.block<2, 3>(0, Estimator::xATT);
  EXPECT_TRUE((dHdAtt.array() != 0.0).all());

  auto dHdGoalPos = H.block<2, 2>(0, Estimator::xGOAL_POS);
  EXPECT_TRUE((dHdGoalPos.array() != 0.0).all());

  auto dHdGoalRho = H.block<2, 1>(0, Estimator::xGOAL_RHO);
  EXPECT_TRUE((dHdGoalRho.array() != 0.0).all());
}

TEST_F(RandomState, goalDepthMeas_Dimension1)
{
  int meas_dims = 0;
  goalDepthMeasModel(xhat, meas_dims, zhat, H);

  const int true_meas_dims = 1;
  EXPECT_EQ(meas_dims, true_meas_dims);
}

TEST_F(RandomState, goalDepthMeas_NonZero)
{
  int meas_dims = 0;
  goalDepthMeasModel(xhat, meas_dims, zhat, H);
  EXPECT_TRUE((zhat.head(meas_dims).array() != 0.).all());
}

TEST_F(RandomState, goalDepthJac_ZerosWhereExpected)
{
  int meas_dims = 0;
  goalDepthMeasModel(xhat, meas_dims, zhat, H);

  // Goal depth model doesnt depend on UAV POS, VEL, DRAG or on GOAL ATT, OMEGA,
  // VEL, or LMS
  auto dHdPos = H.block<1, 3>(0, Estimator::xPOS);
  EXPECT_TRUE(dHdPos.isZero());

  auto dHdVel = H.block<1, 3>(0, Estimator::xVEL);
  EXPECT_TRUE(dHdVel.isZero());

  auto dHdDrag = H.block<1, 1>(0, Estimator::xMU);
  EXPECT_TRUE(dHdDrag.isZero());

  auto dHdGoal =
      H.block<1, Estimator::xZ - Estimator::xGOAL_VEL>(0, Estimator::xGOAL_VEL);
  EXPECT_TRUE(dHdGoal.isZero());

  // Goal meas model does depend on UAV ATT, GOAL POS & RHO
  // All entries should be different from 0
  auto dHdAtt = H.block<1, 3>(0, Estimator::xATT);
  EXPECT_TRUE((dHdAtt.array() != 0.0).all());

  auto dHdGoalPos = H.block<1, 2>(0, Estimator::xGOAL_POS);
  EXPECT_TRUE((dHdGoalPos.array() != 0.0).all());

  auto dHdGoalRho = H.block<1, 1>(0, Estimator::xGOAL_RHO);
  EXPECT_TRUE((dHdGoalRho.array() != 0.0).all());
}

class SimpleState : public ::testing::Test
{
public:
  SimpleState()
  {
    // UAV centered directly above goal
    xhat.setZero();
    xhat(Estimator::xGOAL_RHO) = 0.1;
    xhat(Estimator::xGOAL_LM + 0) = 1.;
    xhat(Estimator::xGOAL_LM + 1) = 0.5;
    xhat(Estimator::xGOAL_LM + 2) = 1.0;
  }
  Estimator::StateVec xhat;
  Estimator::MeasVec zhat;
  Estimator::MeasH H;
};

TEST_F(SimpleState, goalPixelMeas_DirectlyCentered)
{
  int meas_dims = 0;
  goalPixelMeasModel(xhat, meas_dims, zhat, H);

  const double cx = 320.;
  const double cy = 240.;

  EXPECT_EQ(zhat(0), cx);
  EXPECT_EQ(zhat(1), cy);
}

TEST_F(SimpleState, goalPixelJac_CorrectValues)
{
  int meas_dims = 0;
  goalPixelMeasModel(xhat, meas_dims, zhat, H);

  const double fx = 410.;
  const double fy = 420.;
  const double rho_g = xhat(Estimator::xGOAL_RHO);

  EXPECT_DOUBLE_EQ(H(0, Estimator::xATT + 0), fx);
  EXPECT_DOUBLE_EQ(H(0, Estimator::xGOAL_POS + 1), fx * rho_g);
  EXPECT_DOUBLE_EQ(H(1, Estimator::xATT + 1), fy);
  EXPECT_DOUBLE_EQ(H(1, Estimator::xGOAL_POS + 0), -fy * rho_g);
}

TEST_F(SimpleState, goalDepthMeas_Correct)
{
  int meas_dims = 0;
  goalDepthMeasModel(xhat, meas_dims, zhat, H);

  const double true_depth = 10.;
  EXPECT_DOUBLE_EQ(zhat(0), true_depth);
}

TEST_F(SimpleState, goalDepthJac_CorrectValues)
{
  int meas_dims = 0;
  goalDepthMeasModel(xhat, meas_dims, zhat, H);

  const double rho_g = xhat(Estimator::xGOAL_RHO);
  const double true_jac_val = -(1. / rho_g / rho_g);

  EXPECT_DOUBLE_EQ(H(0, Estimator::xGOAL_RHO), true_jac_val);
}

TEST_F(SimpleState, landmarkPixelMeas_Correct)
{
  const int lm_idx = 0;
  int meas_dims = 0;
  landmarkPixelMeasModel(lm_idx, xhat, meas_dims, zhat, H);

  const double true_px = 338.6364;
  const double true_py = 201.8182;

  const double abs_tol = 1e-4;
  EXPECT_NEAR(zhat(0), true_px, abs_tol);
  EXPECT_NEAR(zhat(1), true_py, abs_tol);
}

TEST_F(SimpleState, landmarkPixelJac_CorrectValues)
{
  const int lm_idx = 0;
  int meas_dims = 0;
  landmarkPixelMeasModel(lm_idx, xhat, meas_dims, zhat, H);

  const double true_dpx_dphi = 410.8471;
  const double true_dpx_dtheta = -1.6942;
  const double true_dpx_dpsi = -37.2728;
  const double true_dpx_dgoalx = 0.;
  const double true_dpx_dgoaly = 37.2728;
  const double true_dpx_drho = 169.4215;
  const double true_dpx_dtheta_g = 37.2728;
  const double true_dpx_dlmx = 0.;
  const double true_dpx_dlmy = 37.2728;
  const double true_dpx_dlmz = -1.6942;

  const double true_dpy_dphi = -1.7355;
  const double true_dpy_dtheta = 423.4711;
  const double true_dpy_dpsi = -19.0909;
  const double true_dpy_dgoalx = -38.1818;
  const double true_dpy_dgoaly = 0.;
  const double true_dpy_drho = -347.1074;
  const double true_dpy_dtheta_g = 19.0909;
  const double true_dpy_dlmx = -38.1818;
  const double true_dpy_dlmy = 0.;
  const double true_dpy_dlmz = 3.4711;

  const double abs_tol = 1e-4;
  EXPECT_NEAR(H(0, Estimator::xATT + 0), true_dpx_dphi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 1), true_dpx_dtheta, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 2), true_dpx_dpsi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 0), true_dpx_dgoalx, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 1), true_dpx_dgoaly, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_RHO), true_dpx_drho, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_ATT), true_dpx_dtheta_g, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_LM + 0), true_dpx_dlmx, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_LM + 1), true_dpx_dlmy, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_LM + 2), true_dpx_dlmz, abs_tol);

  EXPECT_NEAR(H(1, Estimator::xATT + 0), true_dpy_dphi, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xATT + 1), true_dpy_dtheta, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xATT + 2), true_dpy_dpsi, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_POS + 0), true_dpy_dgoalx, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_POS + 1), true_dpy_dgoaly, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_RHO), true_dpy_drho, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_ATT), true_dpy_dtheta_g, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_LM + 0), true_dpy_dlmx, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_LM + 1), true_dpy_dlmy, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_LM + 2), true_dpy_dlmz, abs_tol);
}

class ComplexState : public ::testing::Test
{
public:
  ComplexState()
  {
    // UAV centered directly above goal
    xhat.setZero();
    xhat(Estimator::xATT + 0) = 0.1;
    xhat(Estimator::xATT + 1) = -0.3;
    xhat(Estimator::xATT + 2) = 1.2;

    xhat(Estimator::xGOAL_POS + 0) = -0.45;
    xhat(Estimator::xGOAL_POS + 1) = 1.23;
    xhat(Estimator::xGOAL_RHO) = 0.1;

    xhat(Estimator::xGOAL_ATT) = M_PI / 4.;

    xhat(Estimator::xGOAL_LM + 0) = 2.34;
    xhat(Estimator::xGOAL_LM + 1) = -0.75;
    xhat(Estimator::xGOAL_LM + 2) = 1.134;
  }
  Estimator::StateVec xhat;
  Estimator::MeasVec zhat;
  Estimator::MeasH H;
};

TEST_F(ComplexState, goalPixelMeas_Correct)
{
  int meas_dims = 0;
  goalPixelMeasModel(xhat, meas_dims, zhat, H);

  const double true_px = 400.18150;
  const double true_py = 60.84093;

  const double abs_tol = 1e-4;
  EXPECT_NEAR(zhat(0), true_px, abs_tol);
  EXPECT_NEAR(zhat(1), true_py, abs_tol);
}

TEST_F(ComplexState, goalPixelJac_CorrectValues)
{
  int meas_dims = 0;
  goalPixelMeasModel(xhat, meas_dims, zhat, H);

  const double true_dpxdphi = 425.68;
  const double true_dpxdtheta = -16.57;
  const double true_dpxdpsi = -43.71;
  const double true_dpxdposx = -42.01;
  const double true_dpxdposy = 17.68;
  const double true_dpxdrho = 406.50;

  const double true_dpydphi = -35.04;
  const double true_dpydtheta = 502.14;
  const double true_dpydpsi = -41.08;
  const double true_dpydposx = -16.19;
  const double true_dpydposy = -47.05;
  const double true_dpydrho = -505.84;

  const double abs_tol = 1e-2;
  EXPECT_NEAR(H(0, Estimator::xATT + 0), true_dpxdphi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 1), true_dpxdtheta, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 2), true_dpxdpsi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 0), true_dpxdposx, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 1), true_dpxdposy, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_RHO), true_dpxdrho, abs_tol);

  EXPECT_NEAR(H(1, Estimator::xATT + 0), true_dpydphi, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xATT + 1), true_dpydtheta, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xATT + 2), true_dpydpsi, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_POS + 0), true_dpydposx, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_POS + 1), true_dpydposy, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_RHO), true_dpydrho, abs_tol);
}

TEST_F(ComplexState, goalDepthMeas_DirectlyCentered)
{
  int meas_dims = 0;
  goalDepthMeasModel(xhat, meas_dims, zhat, H);

  const double true_depth = 9.13012;

  const double abs_tol = 1e-4;
  EXPECT_NEAR(zhat(0), true_depth, abs_tol);
}

TEST_F(ComplexState, goalDepthJac_CorrectValues)
{
  int meas_dims = 0;
  goalDepthMeasModel(xhat, meas_dims, zhat, H);

  const double true_ddepthdphi = -1.786;
  const double true_ddepthdtheta = 3.875;
  const double true_ddepthdpsi = -0.156;
  const double true_ddepthdposx = -0.0135;
  const double true_ddepthdposy = -0.310;
  const double true_ddepthdrho = -95.056;

  const double abs_tol = 1e-2;
  EXPECT_NEAR(H(0, Estimator::xATT + 0), true_ddepthdphi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 1), true_ddepthdtheta, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 2), true_ddepthdpsi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 0), true_ddepthdposx, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 1), true_ddepthdposy, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_RHO), true_ddepthdrho, abs_tol);
}

TEST_F(ComplexState, landmarkPixelMeas_Correct)
{
  const int lm_idx = 0;
  int meas_dims = 0;
  landmarkPixelMeasModel(lm_idx, xhat, meas_dims, zhat, H);

  const double true_px = 329.1128;
  const double true_py = -15.8178;

  const double abs_tol = 1e-4;
  EXPECT_NEAR(zhat(0), true_px, abs_tol);
  EXPECT_NEAR(zhat(1), true_py, abs_tol);
}

TEST_F(ComplexState, landmarkPixelJac_CorrectValues)
{
  const int lm_idx = 0;
  int meas_dims = 0;
  landmarkPixelMeasModel(lm_idx, xhat, meas_dims, zhat, H);

  const double true_dpx_dphi = 410.2025;
  const double true_dpx_dtheta = 19.4083;
  const double true_dpx_dpsi = -116.6876;
  const double true_dpx_dgoalx = -39.1146;
  const double true_dpx_dgoaly = 14.1791;
  const double true_dpx_drho = -309.6847;
  const double true_dpx_dtheta_g = 74.9573;
  const double true_dpx_dlmx = -17.6320;
  const double true_dpx_dlmy = 37.6843;
  const double true_dpx_dlmz = 3.0968;

  const double true_dpy_dphi = -5.6859;
  const double true_dpy_dtheta = 573.8714;
  const double true_dpy_dpsi = 44.3643;
  const double true_dpy_dgoalx = -15.1424;
  const double true_dpy_dgoaly = -46.1187;
  const double true_dpy_drho = -1211.1447;
  const double true_dpy_dtheta_g = -83.7430;
  const double true_dpy_dlmx = -43.3182;
  const double true_dpy_dlmy = -21.9036;
  const double true_dpy_dlmz = 12.1114;

  const double abs_tol = 1e-4;
  EXPECT_NEAR(H(0, Estimator::xATT + 0), true_dpx_dphi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 1), true_dpx_dtheta, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 2), true_dpx_dpsi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 0), true_dpx_dgoalx, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 1), true_dpx_dgoaly, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_RHO), true_dpx_drho, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_ATT), true_dpx_dtheta_g, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_LM + 0), true_dpx_dlmx, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_LM + 1), true_dpx_dlmy, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_LM + 2), true_dpx_dlmz, abs_tol);

  EXPECT_NEAR(H(1, Estimator::xATT + 0), true_dpy_dphi, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xATT + 1), true_dpy_dtheta, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xATT + 2), true_dpy_dpsi, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_POS + 0), true_dpy_dgoalx, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_POS + 1), true_dpy_dgoaly, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_RHO), true_dpy_drho, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_ATT), true_dpy_dtheta_g, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_LM + 0), true_dpy_dlmx, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_LM + 1), true_dpy_dlmy, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_LM + 2), true_dpy_dlmz, abs_tol);
}

class ComplexStateWithLM2 : public ::testing::Test
{
public:
  ComplexStateWithLM2()
  {
    // UAV centered directly above goal
    xhat.setZero();
    xhat(Estimator::xATT + 0) = 0.1;
    xhat(Estimator::xATT + 1) = -0.3;
    xhat(Estimator::xATT + 2) = 1.2;

    xhat(Estimator::xGOAL_POS + 0) = -0.45;
    xhat(Estimator::xGOAL_POS + 1) = 1.23;
    xhat(Estimator::xGOAL_RHO) = 0.1;

    xhat(Estimator::xGOAL_ATT) = M_PI / 4.;

    xhat(Estimator::xGOAL_LM + 6 + 0) = 2.34;
    xhat(Estimator::xGOAL_LM + 6 + 1) = -0.75;
    xhat(Estimator::xGOAL_LM + 6 + 2) = 1.134;
  }
  Estimator::StateVec xhat;
  Estimator::MeasVec zhat;
  Estimator::MeasH H;
};

TEST_F(ComplexStateWithLM2, landmarkPixelJac_CorrectLocationWithLMIdx)
{
  const int lm_idx = 2;
  const int xLM_IDX = Estimator::xGOAL_LM + 3 * lm_idx;

  int meas_dims = 0;
  landmarkPixelMeasModel(lm_idx, xhat, meas_dims, zhat, H);

  const double true_dpx_dphi = 410.2025;
  const double true_dpx_dtheta = 19.4083;
  const double true_dpx_dpsi = -116.6876;
  const double true_dpx_dgoalx = -39.1146;
  const double true_dpx_dgoaly = 14.1791;
  const double true_dpx_drho = -309.6847;
  const double true_dpx_dtheta_g = 74.9573;
  const double true_dpx_dlmx = -17.6320;
  const double true_dpx_dlmy = 37.6843;
  const double true_dpx_dlmz = 3.0968;

  const double true_dpy_dphi = -5.6859;
  const double true_dpy_dtheta = 573.8714;
  const double true_dpy_dpsi = 44.3643;
  const double true_dpy_dgoalx = -15.1424;
  const double true_dpy_dgoaly = -46.1187;
  const double true_dpy_drho = -1211.1447;
  const double true_dpy_dtheta_g = -83.7430;
  const double true_dpy_dlmx = -43.3182;
  const double true_dpy_dlmy = -21.9036;
  const double true_dpy_dlmz = 12.1114;
  const double abs_tol = 1e-2;

  EXPECT_NEAR(H(0, Estimator::xATT + 0), true_dpx_dphi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 1), true_dpx_dtheta, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xATT + 2), true_dpx_dpsi, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 0), true_dpx_dgoalx, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_POS + 1), true_dpx_dgoaly, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_RHO), true_dpx_drho, abs_tol);
  EXPECT_NEAR(H(0, Estimator::xGOAL_ATT), true_dpx_dtheta_g, abs_tol);
  EXPECT_NEAR(H(0, xLM_IDX + 0), true_dpx_dlmx, abs_tol);
  EXPECT_NEAR(H(0, xLM_IDX + 1), true_dpx_dlmy, abs_tol);
  EXPECT_NEAR(H(0, xLM_IDX + 2), true_dpx_dlmz, abs_tol);

  EXPECT_NEAR(H(1, Estimator::xATT + 0), true_dpy_dphi, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xATT + 1), true_dpy_dtheta, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xATT + 2), true_dpy_dpsi, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_POS + 0), true_dpy_dgoalx, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_POS + 1), true_dpy_dgoaly, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_RHO), true_dpy_drho, abs_tol);
  EXPECT_NEAR(H(1, Estimator::xGOAL_ATT), true_dpy_dtheta_g, abs_tol);
  EXPECT_NEAR(H(1, xLM_IDX + 0), true_dpy_dlmx, abs_tol);
  EXPECT_NEAR(H(1, xLM_IDX + 1), true_dpy_dlmy, abs_tol);
  EXPECT_NEAR(H(1, xLM_IDX + 2), true_dpy_dlmz, abs_tol);
}
