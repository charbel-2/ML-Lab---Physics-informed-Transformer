/**
 * @file   mujoco_ros.cpp
 * @brief  A class for connecting a MuJoCo simulation with ROS2 communication,
 *         extended to publish an end-effector force/torque sensor reading.
 */

#include <mujoco_ros2/mujoco_ros.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>  // For publishing EE wrench

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                         Constructor                                            //
////////////////////////////////////////////////////////////////////////////////////////////////////
MuJoCoROS::MuJoCoROS(const std::string &xmlLocation) : Node("mujoco_node")
{
    // Declare & get parameters for this node
    std::string jointStateTopicName   = this->declare_parameter<std::string>("joint_state_topic_name", "joint_state");
    std::string jointCommandTopicName = this->declare_parameter<std::string>("joint_command_topic_name", "joint_commands");
    std::string controlMode           = this->declare_parameter<std::string>("control_mode", "TORQUE");
    _simFrequency                     = this->declare_parameter<int>("simulation_frequency", 1000);
    int visualisationFrequency        = this->declare_parameter<int>("visualisation_frequency", 20);

    // Load the MuJoCo XML model
    char errorMessage[1000] = "Could not load model.";
    _model = mj_loadXML(xmlLocation.c_str(), nullptr, errorMessage, 1000);
    if (!_model) {
        throw std::runtime_error("[ERROR] [MuJoCo NODE] Problem loading model: " + std::string(errorMessage));
    }
    _model->opt.timestep = 1.0 / static_cast<double>(_simFrequency);

    // Check for force/torque sensors
    ee_force_sensor_id_  = mj_name2id(_model, mjOBJ_SENSOR, "ee_force");
    ee_torque_sensor_id_ = mj_name2id(_model, mjOBJ_SENSOR, "ee_torque");

    if (ee_force_sensor_id_ < 0) {
        RCLCPP_WARN(this->get_logger(), "Force sensor 'ee_force' not found.");
    }
    if (ee_torque_sensor_id_ < 0) {
        RCLCPP_WARN(this->get_logger(), "Torque sensor 'ee_torque' not found.");
    }
    if (ee_force_sensor_id_ >= 0 && ee_torque_sensor_id_ >= 0) {
        ee_wrench_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>("ee_wrench_sim", 10);
    }

    _jointState = mj_makeData(_model);
    _jointStateMessage.name.resize(_model->nq);
    _jointStateMessage.position.resize(_model->nq);
    _jointStateMessage.velocity.resize(_model->nq);
    _jointStateMessage.effort.resize(_model->nq);
    _torqueInput.resize(_model->nq, 0.0);

    for (int i = 0; i < _model->nq; ++i) {
        const char* jname = mj_id2name(_model, mjOBJ_JOINT, i);
        _jointStateMessage.name[i] = jname ? jname : "joint" + std::to_string(i);
    }

    if (controlMode == "POSITION") _controlMode = POSITION;
    else if (controlMode == "VELOCITY") _controlMode = VELOCITY;
    else if (controlMode == "TORQUE") _controlMode = TORQUE;
    else throw std::invalid_argument("[ERROR] [MuJoCo NODE] Unknown control mode: " + controlMode);

    _simTimer = this->create_wall_timer(
        std::chrono::milliseconds(1000 / _simFrequency),
        std::bind(&MuJoCoROS::update_simulation, this));
    _visTimer = this->create_wall_timer(
        std::chrono::milliseconds(1000 / visualisationFrequency),
        std::bind(&MuJoCoROS::update_visualization, this));

    _jointCommandSubscriber = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        jointCommandTopicName, 1,
        std::bind(&MuJoCoROS::joint_command_callback, this, std::placeholders::_1));
    _jointStatePublisher = this->create_publisher<sensor_msgs::msg::JointState>(
        jointStateTopicName, 1);

    if (!glfwInit()) throw std::runtime_error("Failed to initialise GLFW.");
    _window = glfwCreateWindow(1200, 900, "MuJoCo Visualization", nullptr, nullptr);
    if (!_window) throw std::runtime_error("Failed to create GLFW window.");
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&_camera);
    mjv_defaultOption(&_renderingOptions);
    mjv_defaultPerturb(&_perturbation);
    mjr_defaultContext(&_context);
    mjv_makeScene(_model, &_scene, 1000);

    _camera.azimuth      = this->declare_parameter<double>("camera_azimuth", 135);
    _camera.distance     = this->declare_parameter<double>("camera_distance", 2.5);
    _camera.elevation    = this->declare_parameter<double>("camera_elevation", -35);
    _camera.orthographic = this->declare_parameter<bool>("camera_orthographic", true);
    auto focalPoint = this->declare_parameter<std::vector<double>>("camera_focal_point", {0.0, 0.0, 0.5});
    for (int i = 0; i < 3; ++i) _camera.lookat[i] = focalPoint[i];

    glfwMakeContextCurrent(_window);
    mjr_makeContext(_model, &_context, mjFONTSCALE_100);

    RCLCPP_INFO(this->get_logger(),
                "MuJoCo simulation initiated. Publishing joint state on '%s'; subscribing to commands on '%s'.",
                jointStateTopicName.c_str(), jointCommandTopicName.c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    Update the simulation                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::update_simulation()
{
    if (!_model || !_jointState) {
        RCLCPP_ERROR(this->get_logger(), "MuJoCo model or data is not initialized.");
        return;
    }

    if (_controlMode == TORQUE) {
        for (int i = 0; i < _model->nq; ++i) {
            _jointState->ctrl[i] = _torqueInput[i] + _jointState->qfrc_bias[i] - 1 * _jointState->qvel[i];
            _torqueInput[i] = 0.0;
        }
    }

    mj_step(_model, _jointState);

    for (int i = 0; i < _model->nq; ++i) {
        _jointStateMessage.position[i] = _jointState->qpos[i];
        _jointStateMessage.velocity[i] = _jointState->qvel[i];
        _jointStateMessage.effort[i]   = _jointState->actuator_force[i];
    }
    _jointStateMessage.header.stamp = this->get_clock()->now();
    _jointStatePublisher->publish(_jointStateMessage);

    // Publish EE wrench
    if (ee_force_sensor_id_ >= 0 && ee_torque_sensor_id_ >= 0 && ee_wrench_pub_) {
        int adr_f = _model->sensor_adr[ee_force_sensor_id_];
        int adr_t = _model->sensor_adr[ee_torque_sensor_id_];

        geometry_msgs::msg::WrenchStamped msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "ee_link"; // Or your specific site name

        msg.wrench.force.x  = _jointState->sensordata[adr_f + 0];
        msg.wrench.force.y  = _jointState->sensordata[adr_f + 1];
        msg.wrench.force.z  = _jointState->sensordata[adr_f + 2];
        msg.wrench.torque.x = _jointState->sensordata[adr_t + 0];
        msg.wrench.torque.y = _jointState->sensordata[adr_t + 1];
        msg.wrench.torque.z = _jointState->sensordata[adr_t + 2];

        // Optional: Subtract gravity (in world frame)
        int site_id = _model->sensor_objid[ee_force_sensor_id_];
        int body_id = _model->site_bodyid[site_id];
        double mass = _model->body_mass[body_id];
        const double* g = _model->opt.gravity;

        //msg.wrench.force.x  -= mass * g[0];
        //msg.wrench.force.y -= mass * g[1];
        //msg.wrench.force.z  -= mass * g[2];

        ee_wrench_pub_->publish(msg);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    Handle joint commands                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::joint_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if ((int)msg->data.size() != _model->nq) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Received joint command with incorrect size.");
        return;
    }

    switch (_controlMode) {
        case POSITION:
            for (int i = 0; i < _model->nq; ++i) {
                _jointState->ctrl[i] = msg->data[i];
            }
            break;

        case VELOCITY:
            for (int i = 0; i < _model->nq; ++i) {
                _jointState->ctrl[i] += msg->data[i] / (double)_simFrequency;
            }
            break;

        case TORQUE:
            for (int i = 0; i < _model->nq; ++i) {
                _torqueInput[i] = msg->data[i];
            }
            break;

        default:
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "Unknown control mode. Unable to set joint commands.");
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    Update the 3D simulation                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::update_visualization()
{
    if (!_window) return;
    glfwMakeContextCurrent(_window);

    mjv_updateScene(_model, _jointState, &_renderingOptions, nullptr, &_camera, mjCAT_ALL, &_scene);

    int width, height;
    glfwGetFramebufferSize(_window, &width, &height);
    mjrRect viewport = {0, 0, width, height};
    mjr_render(viewport, &_scene, &_context);

    glfwSwapBuffers(_window);
    glfwPollEvents();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                           Destructor                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////
MuJoCoROS::~MuJoCoROS()
{
    if (_jointState) {
        mj_deleteData(_jointState);
        _jointState = nullptr;
    }
    if (_model) {
        mj_deleteModel(_model);
        _model = nullptr;
    }
    mjv_freeScene(&_scene);
    mjr_freeContext(&_context);
    if (_window) {
        glfwDestroyWindow(_window);
        _window = nullptr;
    }
    glfwTerminate();
}
