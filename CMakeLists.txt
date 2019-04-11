cmake_minimum_required(VERSION 2.8.3)
project(landing_sim)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

add_subdirectory(lib/geometry)
add_subdirectory(lib/multirotor_sim)

include_directories(include
	lib
	lib/multirotor_sim/include
	lib/geometry/include
        lib/multirotor_sim/lib/lin_alg_tools/include
	${EIGEN3_INCLUDE_DIRS})

#add_executable(lqr_example_waypoints src/examples/lqr_example_waypoints.cpp)
#target_link_libraries(lqr_example_waypoints lqr multirotor_sim)

add_executable(basic_example src/basic_example.cpp)
target_link_libraries(basic_example multirotor_sim)

add_executable(estimator src/estimator_example.cpp)
target_link_libraries(estimator multirotor_sim)
