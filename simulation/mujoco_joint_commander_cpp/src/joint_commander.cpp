#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <cmath>
#include <vector>

#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>

#include <fstream>
#include <sstream>


class JointCommander : public rclcpp::Node
{
public:
    JointCommander()
        : Node("joint_commander_cpp"), t_(0.0)
    {
        // Publisher for torque commands
        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("joint_commands", 10);

        // Subscriber to joint states
        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states",
            10,
            std::bind(&JointCommander::joint_state_callback, this, std::placeholders::_1));

        // Timer for publishing commands at 100ms interval
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1),
            std::bind(&JointCommander::timer_callback, this));

        // Initialize reference joint positions (example: zero for all joints)
        reference_positions_ = std::vector<double>(7, 0.6);

        // PD gains (tune these as needed)
        Kp_ = 50.0;
        Kd_ = 1.0;

        current_positions_ = std::vector<double>(7, 0.0);
        current_velocities_ = std::vector<double>(7, 0.0);

        RCLCPP_INFO(this->get_logger(), "JointCommander with PD control started.");
    }

private:
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Assuming the joint order matches and 7 joints are published
        if (msg->position.size() >= 7 && msg->velocity.size() >= 7)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                current_positions_[i] = msg->position[i];
                current_velocities_[i] = msg->velocity[i];
            }
        }
    }

    void timer_callback()
    {
        std_msgs::msg::Float64MultiArray torque_msg;
        torque_msg.data.resize(7);

        // PD Control: torque = Kp*(ref_pos - current_pos) - Kd*current_vel
        for (size_t i = 0; i < 7; ++i)
        {
            double error = reference_positions_[i] - current_positions_[i];
            double d_error = -current_velocities_[i];
            torque_msg.data[i] = Kp_ * error + Kd_ * d_error;
            if (torque_msg.data[i]>10)
            {
            	torque_msg.data[i] = 10;
            }
            
        }
        

        publisher_->publish(torque_msg);
    }

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<double> reference_positions_;
    std::vector<double> current_positions_;
    std::vector<double> current_velocities_;

    double Kp_;
    double Kd_;

    double t_;
};

