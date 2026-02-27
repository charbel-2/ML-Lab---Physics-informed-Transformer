cmake_minimum_required(VERSION 3.8)
project(mujoco_joint_commander_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)

# Get MUJOCO_HOME from the environment
if(NOT DEFINED ENV{MUJOCO_HOME})
  message(FATAL_ERROR "Please set MUJOCO_HOME to your MuJoCo install (e.g. ~/.mujoco/mujoco-3.2.0)")
endif()
set(MUJOCO_HOME $ENV{MUJOCO_HOME})
message(STATUS "Using MuJoCo from: ${MUJOCO_HOME}")


# Include and link directories
include_directories(
  ${MUJOCO_HOME}/include
  ${MUJOCO_HOME}/include/mujoco
)
link_directories(
  ${MUJOCO_HOME}/bin
  ${MUJOCO_HOME}/lib
)

# Locate the MuJoCo shared library under MUJOCO_HOME/{bin,lib}
find_library(MUJOCO_LIB
  NAMES mujoco libmujoco.so
  PATHS
    ${MUJOCO_HOME}/bin
    ${MUJOCO_HOME}/lib
  NO_DEFAULT_PATH
)

include_directories(${EIGEN3_INCLUDE_DIR} /usr/local/include/mujoco)

if(NOT MUJOCO_LIB)
  message(FATAL_ERROR
    "Could not find libmujoco (looked in ${MUJOCO_HOME}/bin and /lib)\n"
    "Make sure MUJOCO_HOME is set correctly."
  )
endif()
message(STATUS "Linking against MuJoCo library: ${MUJOCO_LIB}")


add_executable(Cartesian_impedance_controller src/Cartesian_Impedance_controller.cpp)
#add_executable(Cartesian_position_PID_controller src/Cartesian_Position_PID_controller.cpp)

ament_target_dependencies(Cartesian_impedance_controller 
			  
			  rclcpp 
			  std_msgs 
			  sensor_msgs
			  orocos_kdl
			  kdl_parser
			  urdf
			  Eigen3
			  tf2
)

# Link against MuJoCo and system libs
target_link_libraries(Cartesian_impedance_controller
  ${MUJOCO_LIB}   # explicit full path to the .so
  dl                                # dynamic loader
    tf2::tf2         # Link the tf2 library here
)



install(TARGETS Cartesian_impedance_controller DESTINATION lib/${PROJECT_NAME})



# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
