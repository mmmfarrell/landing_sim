cmake_minimum_required(VERSION 2.8.3)
project(landing_sim)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(lib/geometry)
add_subdirectory(lib/multirotor_sim)

include_directories(include
	lib
	lib/multirotor_sim/include
	lib/geometry/include
        lib/multirotor_sim/lib/lin_alg_tools/include
	${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})

add_library(estimator_utils STATIC
  src/measurement_models.cpp
  src/estimator_utils.cpp)
#target_link_libraries(estimator_utils mul)

#add_executable(lqr_example_waypoints src/examples/lqr_example_waypoints.cpp)
#target_link_libraries(lqr_example_waypoints lqr multirotor_sim)

add_executable(basic_example src/basic_example.cpp)
target_link_libraries(basic_example multirotor_sim)

#add_executable(estimator src/estimator.cpp src/estimator_example.cpp)
#target_link_libraries(estimator ${OpenCV_LIBS} multirotor_sim)

add_executable(uav_estimator src/uav_estimator.cpp src/uav_estimator_example.cpp)
target_link_libraries(uav_estimator ${OpenCV_LIBS} multirotor_sim
  estimator_utils)

add_executable(landing_estimator src/landing_estimator.cpp src/landing_estimator_example.cpp)
target_link_libraries(landing_estimator ${OpenCV_LIBS} multirotor_sim
  estimator_utils)

add_executable(test_utils
  src/test_measurements.cpp
  src/test_estimator_utils.cpp)
target_link_libraries(test_utils gtest gtest_main pthread estimator_utils)
