/**
 * @file   mujoco_ros.hpp
 * @brief  A class for connecting a MuJoCo simulation with ROS2 communication,
 *         extended to publish an end-effector force/torque sensor reading.
 */

#ifndef MUJOCO_NODE_H
#define MUJOCO_NODE_H

#include <GLFW/glfw3.h>                              // Graphics Library Framework; for visualization
#include <iostream>                                  // std::cerr, std::cout
#include <mujoco/mujoco.h>                           // MuJoCo dynamic simulation library
#include <rclcpp/rclcpp.hpp>                         // ROS2 C++ libraries
#include <sensor_msgs/msg/joint_state.hpp>           // For publishing joint states
#include <std_msgs/msg/float64_multi_array.hpp>      // For receiving joint commands
#include <geometry_msgs/msg/wrench_stamped.hpp>      // For publishing end-effector wrench

enum ControlMode { POSITION, VELOCITY, TORQUE, UNKNOWN };

/**
 * @brief This class launches both a MuJoCo simulation and a ROS2 node for communication,
 *        and publishes an end-effector force/torque sensor reading when available.
 */
class MuJoCoROS: public rclcpp::Node
{
public:
    /**
     * @brief Constructor.
     * @param xmlLocation Path to the MJCF XML file that defines the MuJoCo model.
     */
    explicit MuJoCoROS(const std::string &xmlLocation);

    /**
     * @brief Destructor.
     */
    ~MuJoCoROS();

private:
    // MuJoCo model and data
    mjModel* _model = nullptr;            ///< MuJoCo model pointer
    mjData*  _jointState = nullptr;       ///< MuJoCo data (positions, velocities, forces, sensordata, etc.)

    // Control inputs
    std::vector<double> _torqueInput;     ///< Stores joint commands in torque mode
    ControlMode _controlMode = UNKNOWN;   ///< POSITION, VELOCITY, or TORQUE mode

    // ROS2 publishers/subscribers/timers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr _jointStatePublisher;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _jointCommandSubscriber;
    rclcpp::TimerBase::SharedPtr _simTimer;
    rclcpp::TimerBase::SharedPtr _visTimer;

    // End-effector force/torque sensor IDs and publisher
    int ee_force_sensor_id_  = -1;   ///< MuJoCo sensor ID for "ee_force" (3D force)
    int ee_torque_sensor_id_ = -1;   ///< MuJoCo sensor ID for "ee_torque" (3D torque)
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ee_wrench_pub_;

    // JointState message buffer
    sensor_msgs::msg::JointState _jointStateMessage;

    // Simulation frequency
    int _simFrequency = 1000;    ///< Rate in Hz to step MuJoCo

    // MuJoCo visualization
    mjvCamera  _camera;            ///< Camera for viewing
    mjvOption  _renderingOptions; ///< Rendering options
    mjvPerturb _perturbation;     ///< For manual interaction
    mjvScene   _scene;            ///< The environment to render
    mjrContext _context;          ///< MuJoCo rendering context
    GLFWwindow *_window = nullptr;///< GLFW window pointer

    /**
     * @brief Update the MuJoCo simulation one step and publish joint state (and EE wrench if available).
     */
    void update_simulation();

    /**
     * @brief Update the MuJoCo visualization (rendering).
     */
    void update_visualization();

    /**
     * @brief Callback to handle incoming joint commands.
     * @param msg The message containing joint commands.
     */
    void joint_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
};

#endif  // MUJOCO_NODE_H
