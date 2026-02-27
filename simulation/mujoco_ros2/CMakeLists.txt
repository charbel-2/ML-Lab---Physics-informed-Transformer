####################################################################################################
#                                         PROJECT INFORMATION                                      #
####################################################################################################
cmake_minimum_required(VERSION 3.5)
project(mujoco_ros2)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

####################################################################################################
#                                          DEPENDENCIES                                            #
####################################################################################################
find_package(ament_cmake REQUIRED)                                                                  # So ROS2 can compile?
find_package(glfw3 REQUIRED)                                                                        # Graphics Library FrameWork
find_package(rclcpp REQUIRED)                                                                       # ROS2 C++ libraries
find_package(sensor_msgs REQUIRED)                                                                  # sensor_msgs::msg::JointState
find_package(std_msgs REQUIRED)                                                                     # std_msgs::msg::Float64MultiArray

# MuJoCo
set(MUJOCO_PATH "$ENV{HOME}/mujoco/mujoco-3.2.0")
include_directories(${MUJOCO_PATH}/include)
include_directories(${MUJOCO_PATH}/include/mujoco)
link_directories(${MUJOCO_PATH}/lib)

#set(MUJOCO_PATH "/opt/mujoco/mujoco-3.2.0") # UPDATE THIS TO YOUR MUJOCO PATH
#include_directories(include)                                                                        # Local header files
#include_directories(${MUJOCO_PATH}/include/mujoco)                                                  # MuJoCo header files
#include_directories(${MUJOCO_PATH}/include)                                                         # Additional MuJoCo header files
#link_directories(${MUJOCO_PATH}/lib)                                                                # Location of MuJoCo libraries

####################################################################################################
#                                           EXECUTABLES                                            #
####################################################################################################

# so that <mujoco_ros2/mujoco_ros.hpp> is found:
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(mujoco_node src/nodes/mujoco_node.cpp
                           src/mujoco_ros.cpp)                                                      # Define new executable, list source files
    
target_link_libraries(mujoco_node mujoco glfw)                                                      # Link GLFW libraries to the executable
                                     
ament_target_dependencies(mujoco_node rclcpp sensor_msgs std_msgs)                                  # List ROS2 dependencies for this executable


####################################################################################################
#                                           INSTALLATION                                           #
####################################################################################################

install(TARGETS mujoco_node
        DESTINATION lib/${PROJECT_NAME})                                                            # Install so ROS2 can find it

install(DIRECTORY launch
                  model
        DESTINATION share/${PROJECT_NAME})                                                          # Install so ROS2 can find them
        
ament_package()

