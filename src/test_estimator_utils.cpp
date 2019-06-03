#include <gtest/gtest.h>

#include <estimator_utils.h>

template <typename T>
void EXPECT_NEAR_MAT(T mat1, T mat2, double tol)
{
  ASSERT_EQ(mat1.rows(), mat2.rows());
  ASSERT_EQ(mat1.cols(), mat2.cols());

  for (unsigned int i = 0; i < mat1.rows(); i++)
  {
    for (unsigned int j = 0; j < mat1.cols(); j++)
    {
      EXPECT_NEAR(mat1(i, j), mat2(i, j), tol);
    }
  }

}

template <typename T>
void EXPECT_EQ_MAT(T mat1, T mat2)
{
  ASSERT_EQ(mat1.rows(), mat2.rows());
  ASSERT_EQ(mat1.cols(), mat2.cols());

  for (unsigned int i = 0; i < mat1.rows(); i++)
  {
    for (unsigned int j = 0; j < mat1.cols(); j++)
    {
      EXPECT_EQ(mat1(i, j), mat2(i, j));
    }
  }

}

TEST(R_I_b, zeroRPY_ReturnsIdentity)
{
  double roll = 0.;
  double pitch = 0.;
  double yaw = 0.;

  const Eigen::Matrix3d R_I_b = rotmItoB(roll, pitch, yaw);

  EXPECT_EQ(R_I_b, Eigen::Matrix3d::Identity());
}

TEST(R_I_b, onlyYaw_ReturnsCorrect)
{
  double roll = 0.;
  double pitch = 0.;
  double yaw = M_PI;

  const Eigen::Matrix3d R_I_b = rotmItoB(roll, pitch, yaw);

  Eigen::Matrix3d true_R_I_b = Eigen::Matrix3d::Identity();
  true_R_I_b(0, 0) = -1.;
  true_R_I_b(1, 1) = -1.;

  double tol = 1e-5;
  EXPECT_NEAR_MAT(R_I_b, true_R_I_b, tol);
}

TEST(R_I_b, RPY_ReturnsCorrect)
{
  double roll = 1.45;
  double pitch = -0.32;
  double yaw = 0.567;

  const Eigen::Matrix3d R_I_b = rotmItoB(roll, pitch, yaw);

  Eigen::Vector3d vec_I(1., 1., 1.);
  Eigen::Vector3d vec_b = R_I_b * vec_I;

  Eigen::Vector3d true_vec_b(1.62510, 0.54811, -0.24213);

  double tol = 1e-5;
  EXPECT_NEAR_MAT(vec_b, true_vec_b, tol);
}
