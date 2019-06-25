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

class SimpleState : public ::testing::Test
{
public:
  SimpleState()
  {
    // UAV centered directly above goal
    xhat.setZero();
    xhat(Estimator::xGOAL_RHO) = 0.1;
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
  }
  Estimator::StateVec xhat;
  Estimator::MeasVec zhat;
  Estimator::MeasH H;
};

TEST_F(ComplexState, goalPixelMeas_DirectlyCentered)
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
