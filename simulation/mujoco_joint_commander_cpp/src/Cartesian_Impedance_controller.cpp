#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>    
#include <geometry_msgs/msg/accel_stamped.hpp>    // <--- for acceleration publishing
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>       // <-- for velocity IK
#include <kdl/jntarray.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/frames.hpp>
#include <kdl/chaindynparam.hpp>

#include <urdf/model.h>
#include <fstream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>  // for quaternion operations

// MuJoCo 3.x
extern "C" {
  #include <mujoco.h>
}

#include <deque>  // For buffering last samples

class JointSpacePositionController : public rclcpp::Node {
public:
  JointSpacePositionController()
    : Node("joint_space_position_controller"),
      have_raw_wrench_(false),
      first_tau_ext_(true),
      mj_model_(nullptr),
      mj_data_(nullptr),
      panda_wrist_body_id_(-1),
      freeze_(false),
      first_cmd_(true),
      have_last_desired_pose_(false),
      have_last_actual_ee_velocity_(false),
      have_last_desired_ee_velocity_(false),
      first_actual_vel_(true),
      first_actual_acc_(true),
      have_optitrack_pose_(false),
      have_future_optitrack_pose_(false),
      seq_length_(352)      
  {
    // Parameters
    declare_parameter<std::string>("urdf_path",
      "/home/polimi-panda/ros2_ws/src/robot_descriptions/urdf/panda.urdf");
    declare_parameter<std::string>("base_link", "panda_link0"); // if using panda.urdf: panda_link0 or panda_link8
    declare_parameter<std::string>("tip_link",  "panda_hand");

    declare_parameter<double>("ewma_alpha", 0.99);     // joint-space filter alpha
    declare_parameter<double>("damp_lambda", 0.05);    // damping for mapping back

    declare_parameter<std::string>("mujoco_xml_path",
      "/home/polimi-panda/mujoco_models/franka_panda/panda_nohand.xml");

    // New parameter for commanded torque filtering
    declare_parameter<double>("cmd_ewma_alpha", 0.9);  // closer to 1 = smoother/slower response

    // Parameters for filtering actual EE velocity and acceleration
    declare_parameter<double>("ee_vel_ewma_alpha", 0.8);    // smoothing actual EE velocity
    declare_parameter<double>("ee_accel_ewma_alpha", 0.8);  // smoothing actual EE acceleration

    // Fill parameters in place
    get_parameter("urdf_path", urdf_path_);
    get_parameter("base_link", base_link_);
    get_parameter("tip_link", tip_link_);
    get_parameter("ewma_alpha", ewma_alpha_);
    get_parameter("damp_lambda", damp_lambda_);
    get_parameter("mujoco_xml_path", mujoco_xml_path_);
    get_parameter("cmd_ewma_alpha", cmd_ewma_alpha_);
    get_parameter("ee_vel_ewma_alpha", ee_vel_ewma_alpha_);
    get_parameter("ee_accel_ewma_alpha", ee_accel_ewma_alpha_);

    // Load URDF & init KDL
    if (!loadURDFFile(urdf_path_, urdf_xml_)) {
      RCLCPP_ERROR(get_logger(), "Failed to load URDF from '%s'", urdf_path_.c_str());
      rclcpp::shutdown(); return;
    }
    if (!initializeKDL(urdf_xml_)) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize KDL from URDF");
      rclcpp::shutdown(); return;
    }

    // get number of joints
    size_t nj = kdl_chain_.getNrOfJoints();

    // Resize arrays
    current_q_ = KDL::JntArray(nj);
    current_dq_ = KDL::JntArray(nj);
    desired_q_ = KDL::JntArray(nj);
    prev_dq_ = KDL::JntArray(nj);
    qddot_.resize(nj);
    jacobian_ = KDL::Jacobian(nj);
    q_kdl_.resize(nj);
    dyn_param_ = std::make_shared<KDL::ChainDynParam>(kdl_chain_, KDL::Vector(0,0, -9.81));
    tau_g_ = KDL::JntArray(nj);

    frozen_q_ = KDL::JntArray(nj);

    // Initialize filters
    // ewma_alpha_ = std::clamp(ewma_alpha_, 0.0, 1.0);
    damp_lambda_ = std::max(0.0, damp_lambda_);
    filtered_tau_ext_.setZero(nj);
    have_raw_wrench_ = false;
    first_tau_ext_ = true;

    // Initialize commanded torque filter
    cmd_ewma_alpha_ = std::clamp(cmd_ewma_alpha_, 0.0, 1.0);
    filtered_cmd_.setZero(nj);
    first_cmd_ = true;

    // PID parameters
    // Kp_joint_.setConstant(500.0);
    // Kd_joint_.setConstant(20.0);
    // Ki_joint_.setConstant(40.0);
    delta_tau_max_ = 20.0;
    integral_joint_error_.setZero(nj);
    last_torque_.setZero(nj);

    // control gains from cartesian impedance controller
    Kp_joint_(0) = 1000;  // to double. 
    Kp_joint_(1) = 750;  // 750.
    Kp_joint_(2) = 750;  // 750.
    Kp_joint_(3) = 750;  // 750.
    Kp_joint_(4) = 200.*2.5;
    Kp_joint_(5) = 100.*2.5;
    Kp_joint_(6) = 50. *2.5;
    //all values were half before
    Kd_joint_(0) = 32. *2;   // 32.
    Kd_joint_(1) = 40. *2;   // 40.
    Kd_joint_(2) = 30. *2;   // 30.
    Kd_joint_(3) = 45  *2;  // 45.
    Kd_joint_(4) = 20. *2;   // 20.
    Kd_joint_(5) = 20. *2;   // 20.
    Kd_joint_(6) = 10. *2;    // 10. rimasto 10 nella prova

    Ki_joint_(0) = 15 ;  // 15.
    Ki_joint_(1) = 15.; // 15.
    Ki_joint_(2) = 20.;  // 20.
    Ki_joint_(3) = 40.;  // 40.
    Ki_joint_(4) = 40.;  // 40.
    Ki_joint_(5) = 30.;  // 30. rimasto 30 nella prova
    Ki_joint_(6) = 40.;  // 40. rimasto 40 nelòla prova

    // Init previous dq/time
    prev_time_ = now();
    for (size_t i = 0; i < nj; ++i) {
      prev_dq_(i) = 0.0;
      qddot_(i)   = 0.0;
    }
    // Initialize joint-velocity buffer
    // We will store up to 3 entries of (time, joint velocities) to compute central differences
    dq_buffer_.clear();

    // Initialize buffers for EE velocity/acceleration
    last_actual_ee_vel_buffer_.clear();

    // Initialize buffers for desired EE pose -> velocity -> acceleration
    last_desired_pose_buffer_.clear();
    last_desired_vel_buffer_.clear();

    // Initialize buffers for future desired EE pose -> velocity -> acceleration
    last_future_desired_pose_buffer_.clear();
    last_future_desired_vel_buffer_.clear();

    // Initialize filtering state for actual EE velocity/acceleration
    filtered_actual_ee_vel_.setZero();
    filtered_actual_ee_accel_.setZero();
    first_actual_vel_ = true;
    first_actual_acc_ = true;

    // Subscribers / Publishers

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_state", 10,
      std::bind(&JointSpacePositionController::jointStateCallback, this, std::placeholders::_1)
    );
    desired_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/opt_traj", 10,
      // optitrack_pose // test with mpc for now
      std::bind(&JointSpacePositionController::desiredPoseCallback, this, std::placeholders::_1)
    );


    ee_wrench_sub_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/ee_wrench_sim", 10,
      std::bind(&JointSpacePositionController::eeWrenchCallback, this, std::placeholders::_1)
    );
    
    // New subscription & publisher for future desired EE pose
    future_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/opt_traj", 10,
      std::bind(&JointSpacePositionController::futureDesiredPoseCallback, this, std::placeholders::_1)
    );
    

    // /ee_contact_wrench includes the cleaned "interaction" external forces at ee
    // /ee_contact_wrench_filtered is /ee_contact_wrench with a filter ---> this is the one to use for controllers
    // For higher level controllers please subscribe to topics from this code:

    // actual position ---> actual_ee_pose (PoseStamped)
    // actual velocity --> actual_ee_velocity (TwistStamped)
    // actual acceleration --> actual_ee_acceleration (AccelStamped)

    // desired posittion ---> desired_ee_pose (PoseStamped)
    // desired velocity --> desired_ee_velocity (TwistStamped)

    // future desired positon --> future_desired_ee_pose (PoseStamped)
    // future desired velocity --> future_desired_ee_velocity (TwistStamped)
    // future desired acceleration --> future_desired_ee_acceleration (AccelStamped)

    // interaction force --> ee_contact_wrench_filtered (WrenchStamped)

    // impedance_reference_pose --> new ee pose from impedance (PoseStamped)


    // IMPORTANT: IF I PUBLISH JOINT_COMMANDS FROM THE TRANSFORMER NODE, THE TRANSFORMER WORKS WELL, BUT WE CAN'T MOVE THE ROBOT 


    joint_command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("joint_commands", 1);

    // ee_contact_wrench_pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
    //   "/ee_contact_wrench", 10);
    
    ee_contact_wrench_filtered_pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
      "/ee_contact_wrench_filtered", 10);

    desired_ee_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "desired_ee_pose", 10); // test with mpc
      
    actual_ee_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "actual_ee_pose", 10);

    actual_ee_velocity_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
      "actual_ee_velocity", 10);

    // === ADD: publisher for actual EE acceleration ===
    actual_ee_acceleration_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>(
      "actual_ee_acceleration", 10);
    // Initialize flag
    have_last_actual_ee_velocity_ = false;
    // ==================================================

    impedance_ref_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "impedance_reference_pose", 10);

    future_desired_ee_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "future_desired_ee_pose", 10);

    // === ADDITION: publisher for desired EE velocity ===
    desired_ee_velocity_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
      "desired_ee_velocity", 10);

    desired_ee_acceleration_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>(
      "desired_ee_acceleration", 10);
    // ==========================================================

    future_desired_ee_velocity_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
      "future_desired_ee_velocity", 10);
    future_desired_ee_acceleration_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>(
      "future_desired_ee_acceleration", 10);
    have_last_desired_ee_velocity_ = false;
    // ==========================================================
    // Create and publish sequences

    seq_ee_contact_wrench_filtered_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "/ee_contact_wrench_filtered_seq", 10);

    seq_actual_ee_pose_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "actual_ee_pose_seq", 10);

    seq_actual_ee_velocity_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "actual_ee_velocity_seq", 10);

    seq_actual_ee_acceleration_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "actual_ee_acceleration_seq", 10);

    seq_desired_ee_pose_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "desired_ee_pose_seq", 10);

    seq_desired_ee_velocity_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "desired_ee_velocity_seq", 10);

    seq_desired_ee_acceleration_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "desired_ee_acceleration_seq", 10);

    seq_future_desired_ee_pose_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "future_desired_ee_pose_seq", 10);

    seq_future_desired_ee_velocity_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "future_desired_ee_velocity_seq", 10);

    seq_future_desired_ee_acceleration_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "future_desired_ee_acceleration_seq", 10);

    // virtual impedance parameters:
    // Use values from georges's code to make the robot jump or oscillate like a spring, maybe to test the controller
    mass_imp_ = 15.0;               // 5
    inertia_imp_ = 5.0;  // 5
    stiffness_trans_ = 200.0; // 100
    stiffness_rot_ = 200.0; // 100
    damping_factor_trans_ = 5.0; // 1
    damping_factor_rot_ = 5.0; // 1
    
    // impedance matrices definition
    M_cart_.setZero();
    M_inv_.setZero();
    M_cart_(0,0) = mass_imp_;
    M_cart_(1,1) = mass_imp_;
    M_cart_(2,2) = mass_imp_;
    M_cart_(3,3) = inertia_imp_;
    M_cart_(4,4) = inertia_imp_;
    M_cart_(5,5) = inertia_imp_;
    M_inv_ = M_cart_.inverse();
    // K_cart_ diagonal
    K_cart_.setZero();
    K_cart_(0,0) = stiffness_trans_;
    K_cart_(1,1) = stiffness_trans_;
    K_cart_(2,2) = stiffness_trans_;
    K_cart_(3,3) = stiffness_rot_;
    K_cart_(4,4) = stiffness_rot_;
    K_cart_(5,5) = stiffness_rot_;
    // D_cart_ via critical damping: D = 2 * sqrt(K * M) * damping_factor
    D_cart_.setZero();
    double d_t = 2.0 * damping_factor_trans_ * std::sqrt(stiffness_trans_ * mass_imp_);
    D_cart_(0,0) = d_t;
    D_cart_(1,1) = d_t;
    D_cart_(2,2) = d_t;
    double d_r = 2.0 * damping_factor_rot_ * std::sqrt(stiffness_rot_ * inertia_imp_);
    D_cart_(3,3) = d_r;
    D_cart_(4,4) = d_r;
    D_cart_(5,5) = d_r;

    // Initialize offset state
    offset_pos_.setZero();
    offset_vel_.setZero();

    // Initialize stored poses
    desired_pos_.setZero();
    desired_rpy_.setZero();
    actual_pos_.setZero();
    actual_rpy_.setZero();
    last_dt_ = 0.001; // default
    // ==========================================================

    RCLCPP_INFO(get_logger(), "Cartesian impedance controller started: uses EE wrench mapping for external force estimation.");

    // MuJoCo initialization and loading the mj_model_
    {
      char error_buffer[1000] = "Could not load MuJoCo model";
      mj_model_ = mj_loadXML(mujoco_xml_path_.c_str(), nullptr, error_buffer, 1000);
      if (!mj_model_) {
        RCLCPP_ERROR(this->get_logger(), "MuJoCo XML load error: %s", error_buffer);
        rclcpp::shutdown();
        return;
      }
      mj_data_ = mj_makeData(mj_model_);
      if (!mj_data_) {
        RCLCPP_ERROR(this->get_logger(), "Failed to allocate mjData.");
        mj_deleteModel(mj_model_);
        rclcpp::shutdown();
        return;
      }
      panda_wrist_body_id_ = mj_name2id(mj_model_, mjOBJ_BODY, "ee_tool"); // select "link7" to not consider the ee tool
      if (panda_wrist_body_id_ == -1) {
        RCLCPP_ERROR(this->get_logger(), "Could not find MuJoCo body named 'ee_tool'.");
        mj_deleteData(mj_data_);
        mj_deleteModel(mj_model_);
        rclcpp::shutdown();
        return;
      }
      RCLCPP_INFO(get_logger(), "MuJoCo model loaded successfully, body 'link7' id=%d", panda_wrist_body_id_);
    }
  }

private:

  bool have_optitrack_pose_;
  bool have_future_optitrack_pose_;

  // Load URDF into string
  bool loadURDFFile(const std::string &path, std::string &urdf_xml) {
    std::ifstream file(path);
    if (!file.is_open()) {
      RCLCPP_ERROR(get_logger(), "Cannot open URDF file: %s", path.c_str());
      return false;
    }
    urdf_xml.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return true;
  }

  // Initialize KDL chain
  bool initializeKDL(const std::string &urdf_xml) {
    urdf::Model model;
    if (!model.initString(urdf_xml)) {
      RCLCPP_ERROR(get_logger(), "URDF model initString failed");
      return false;
    }
    KDL::Tree tree;
    if (!kdl_parser::treeFromUrdfModel(model, tree)) {
      RCLCPP_ERROR(get_logger(), "kdl_parser::treeFromUrdfModel failed");
      return false;
    }
    if (!tree.getChain(base_link_, tip_link_, kdl_chain_)) {
      RCLCPP_ERROR(get_logger(), "Failed to get KDL chain from %s to %s",
                   base_link_.c_str(), tip_link_.c_str());
      return false;
    }
    // jac_solver - computes 6xn_j Jacobian
    // fk_solver = forward kinematics 
    // ik_solver = inverse kinematics --> switch from cartesian to joint space

    fk_solver_  = std::make_shared<KDL::ChainFkSolverPos_recursive>(kdl_chain_);
    jac_solver_ = std::make_shared<KDL::ChainJntToJacSolver>(kdl_chain_);
    ik_solver_  = std::make_shared<KDL::ChainIkSolverPos_LMA>(kdl_chain_);
    // Initialize velocity IK solver (pseudoinverse):
    vel_ik_solver_ = std::make_shared<KDL::ChainIkSolverVel_pinv>(kdl_chain_);

    RCLCPP_INFO(get_logger(), "Initialized KDL chain with %d joints", (int)kdl_chain_.getNrOfJoints());
    return true;
  }

  // Update joint acceleration by finite difference with buffer
  void updateAcceleration(const rclcpp::Time &stamp) {
    // Maintain dq_buffer_ of up to 3 entries: (time, joint velocities)
    // Push current sample
    KDL::JntArray dq_copy = current_dq_;
    dq_buffer_.push_back({stamp, dq_copy});
    // If more than 3, pop oldest
    if (dq_buffer_.size() > 3) {
      dq_buffer_.pop_front();
    }

    // Compute acceleration:
    if (dq_buffer_.size() >= 2) {
      if (dq_buffer_.size() == 2) {
        // Forward difference
        const auto &e0 = dq_buffer_[0];
        const auto &e1 = dq_buffer_[1];
        double dt = 0.001;
        if (dt > 1e-6) {
          for (size_t i = 0; i < qddot_.size(); ++i) {
            qddot_(i) = (e1.second(i) - e0.second(i)) / dt;
          }
          last_dt_ = dt;
        } else {
          // dt too small: skip acceleration update
          RCLCPP_WARN(get_logger(), "updateAcceleration: dt too small or non-positive, skipping joint acceleration update");
        }
      } else {
        // size == 3: central difference: (v[n] - v[n-2]) / (t[n] - t[n-2])
        const auto &e0 = dq_buffer_[0];
        const auto &e2 = dq_buffer_[2];
        double dt_total = 0.002;
        if (dt_total > 1e-6) {
          for (size_t i = 0; i < qddot_.size(); ++i) {
            qddot_(i) = (e2.second(i) - e0.second(i)) / dt_total;
          }
          last_dt_ = dt_total / 2.0; // approximate per-step dt
        } else {
          RCLCPP_WARN(get_logger(), "updateAcceleration: total dt too small or non-positive, skipping joint acceleration update");
        }
      }
    }
    // else size<2: cannot compute acceleration yet; leave qddot_ as is
  }

  // Saturate torque rate
  Eigen::Matrix<double,7,1> saturateTorqueRate(const Eigen::Matrix<double,7,1> &tau_cmd) {
    Eigen::Matrix<double,7,1> tau_out;
    for (size_t i=0; i<7; ++i) {
      double d = tau_cmd(i) - last_torque_(i);
      d = std::clamp(d, -delta_tau_max_, delta_tau_max_);
      tau_out(i) = last_torque_(i) + d;
    }
    return tau_out;
  }

  // Quaternion to RPY
  void quaternionToRPY(double qx,double qy,double qz,double qw,double &roll,double &pitch,double &yaw) {
    Eigen::Quaterniond q(qw,qx,qy,qz);
    auto R = q.toRotationMatrix();
    roll  = std::atan2(R(2,1), R(2,2));
    pitch = std::atan2(-R(2,0), std::sqrt(R(2,1)*R(2,1) + R(2,2)*R(2,2)));
    yaw   = std::atan2(R(1,0), R(0,0));
  }
  // New callback for future desired EE pose
  void futureDesiredPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    geometry_msgs::msg::PoseStamped future_msg;
    // copy header and pose directly
    
    future_msg.header = msg->header;
    future_msg.pose = msg->pose;
    future_desired_ee_pose_pub_->publish(future_msg);

    std::vector<float> seq_future_desired_pose = {
      static_cast<float>(msg->pose.position.x),
      static_cast<float>(msg->pose.position.y),
      static_cast<float>(msg->pose.position.z),
    };

    future_desired_pose_buffer.push_back(seq_future_desired_pose);

    if (future_desired_pose_buffer.size() > seq_length_) future_desired_pose_buffer.pop_front();

    if (future_desired_pose_buffer.size() == seq_length_)
    {
      std_msgs::msg::Float32MultiArray seq_future_desired_pose_msg;

      seq_future_desired_pose_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
      seq_future_desired_pose_msg.layout.dim[0].label = "sequence";
      seq_future_desired_pose_msg.layout.dim[0].size = seq_length_*3;

      // Flatten and copy
      for (const auto & sample: future_desired_pose_buffer)
      seq_future_desired_pose_msg.data.insert(seq_future_desired_pose_msg.data.end(), sample.begin(), sample.end());

      seq_future_desired_ee_pose_pub_->publish(seq_future_desired_pose_msg);
    }

    have_future_optitrack_pose_ = true;

    // Buffer future-desired poses for velocity computation
    last_future_desired_pose_buffer_.push_back(*msg);
    if (last_future_desired_pose_buffer_.size() > 3) {
      last_future_desired_pose_buffer_.pop_front();
    }

    // === ADDITION: compute and publish future desired EE velocity ===
    bool computed_fut_vel = false;
    geometry_msgs::msg::TwistStamped fut_vel_msg;
    fut_vel_msg.header = msg->header;

    if (last_future_desired_pose_buffer_.size() >= 2) {
      if (last_future_desired_pose_buffer_.size() == 2) {
        // Forward difference between 0 and 1
        const auto &p0 = last_future_desired_pose_buffer_[0];
        const auto &p1 = last_future_desired_pose_buffer_[1];
        rclcpp::Time t0(p0.header.stamp);
        rclcpp::Time t1(p1.header.stamp);
        double dt = 0.001;
        if (dt > 1e-6) {
          // Linear velocity
          fut_vel_msg.twist.linear.x = (p1.pose.position.x - p0.pose.position.x)/dt;
          fut_vel_msg.twist.linear.y = (p1.pose.position.y - p0.pose.position.y)/dt;
          fut_vel_msg.twist.linear.z = (p1.pose.position.z - p0.pose.position.z)/dt;
          // Angular velocity: quaternion difference over dt
          Eigen::Quaterniond q_prev(
            p0.pose.orientation.w,
            p0.pose.orientation.x,
            p0.pose.orientation.y,
            p0.pose.orientation.z
          );
          Eigen::Quaterniond q_cur(
            p1.pose.orientation.w,
            p1.pose.orientation.x,
            p1.pose.orientation.y,
            p1.pose.orientation.z
          );
          Eigen::Quaterniond q_delta = q_cur * q_prev.inverse();
          q_delta.normalize();
          double angle = 2.0 * std::acos(std::clamp(q_delta.w(), -1.0, 1.0));
          Eigen::Vector3d axis;
          double s = std::sqrt(1.0 - q_delta.w()*q_delta.w());
          if (s < 1e-6) {
            axis.setZero();
          } else {
            axis = Eigen::Vector3d(q_delta.x(), q_delta.y(), q_delta.z()) / s;
          }
          Eigen::Vector3d ang_vel = axis * (angle / dt);
          fut_vel_msg.twist.angular.x = ang_vel.x();
          fut_vel_msg.twist.angular.y = ang_vel.y();
          fut_vel_msg.twist.angular.z = ang_vel.z();
          computed_fut_vel = true;
        } else {
          RCLCPP_WARN(get_logger(), "futureDesiredPoseCallback: dt too small or non-positive, skipping future desired velocity publish");
        }
      } else {
        // size == 3: central difference between 0 and 2
        const auto &p0 = last_future_desired_pose_buffer_[0];
        const auto &p2 = last_future_desired_pose_buffer_[2];
        rclcpp::Time t0(p0.header.stamp);
        rclcpp::Time t2(p2.header.stamp);
        double dt_total = 0.001;
        if (dt_total > 1e-6) {
          // Linear velocity
          fut_vel_msg.twist.linear.x = (p2.pose.position.x - p0.pose.position.x)/dt_total;
          fut_vel_msg.twist.linear.y = (p2.pose.position.y - p0.pose.position.y)/dt_total;
          fut_vel_msg.twist.linear.z = (p2.pose.position.z - p0.pose.position.z)/dt_total;
          // Angular velocity: quaternion difference over dt_total
          Eigen::Quaterniond q_prev(
            p0.pose.orientation.w,
            p0.pose.orientation.x,
            p0.pose.orientation.y,
            p0.pose.orientation.z
          );
          Eigen::Quaterniond q_cur(
            p2.pose.orientation.w,
            p2.pose.orientation.x,
            p2.pose.orientation.y,
            p2.pose.orientation.z
          );
          Eigen::Quaterniond q_delta = q_cur * q_prev.inverse();
          q_delta.normalize();
          double angle = 2.0 * std::acos(std::clamp(q_delta.w(), -1.0, 1.0));
          Eigen::Vector3d axis;
          double s = std::sqrt(1.0 - q_delta.w()*q_delta.w());
          if (s < 1e-6) {
            axis.setZero();
          } else {
            axis = Eigen::Vector3d(q_delta.x(), q_delta.y(), q_delta.z()) / s;
          }
          Eigen::Vector3d ang_vel = axis * (angle / dt_total);
          fut_vel_msg.twist.angular.x = ang_vel.x();
          fut_vel_msg.twist.angular.y = ang_vel.y();
          fut_vel_msg.twist.angular.z = ang_vel.z();
          computed_fut_vel = true;
        } else {
          RCLCPP_WARN(get_logger(), "futureDesiredPoseCallback: total dt too small, skipping future desired velocity publish");
        }
      }
    }

    if (computed_fut_vel) {
      // Publish on future_desired_ee_velocity
      future_desired_ee_velocity_pub_->publish(fut_vel_msg);

      std::vector<float> seq_future_desired_vel = {
        static_cast<float>(fut_vel_msg.twist.linear.x),
        static_cast<float>(fut_vel_msg.twist.linear.y),
        static_cast<float>(fut_vel_msg.twist.linear.z),
      };

      future_desired_vel_buffer.push_back(seq_future_desired_vel);

      if (future_desired_vel_buffer.size() > seq_length_) future_desired_vel_buffer.pop_front();

      if (future_desired_vel_buffer.size() == seq_length_)
      {
        std_msgs::msg::Float32MultiArray seq_future_desired_vel_msg;

        seq_future_desired_vel_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
        seq_future_desired_vel_msg.layout.dim[0].label = "sequence";
        seq_future_desired_vel_msg.layout.dim[0].size = seq_length_*3;

        // Flatten and copy
        for (const auto & sample: future_desired_vel_buffer)
        seq_future_desired_vel_msg.data.insert(seq_future_desired_vel_msg.data.end(), sample.begin(), sample.end());

        seq_future_desired_ee_velocity_pub_->publish(seq_future_desired_vel_msg);
      }

      // Buffer this future desired velocity for future acceleration computation
      last_future_desired_vel_buffer_.push_back(fut_vel_msg);
      if (last_future_desired_vel_buffer_.size() > 3) last_future_desired_vel_buffer_.pop_front();

      // === ADDITION: compute and publish future desired EE acceleration ===
      bool computed_fut_acc = false;
      geometry_msgs::msg::AccelStamped acc_msg;
      acc_msg.header = fut_vel_msg.header;
      if (last_future_desired_vel_buffer_.size() >= 2) {
        if (last_future_desired_vel_buffer_.size() == 2) {
          // Forward difference
          const auto &v0 = last_future_desired_vel_buffer_[0];
          const auto &v1 = last_future_desired_vel_buffer_[1];
          rclcpp::Time t0(v0.header.stamp);
          rclcpp::Time t1(v1.header.stamp);
          double dtv = 0.001;
          if (dtv > 1e-6) {
            acc_msg.accel.linear.x  = (v1.twist.linear.x  - v0.twist.linear.x ) / dtv;
            acc_msg.accel.linear.y  = (v1.twist.linear.y  - v0.twist.linear.y ) / dtv;
            acc_msg.accel.linear.z  = (v1.twist.linear.z  - v0.twist.linear.z ) / dtv;
            acc_msg.accel.angular.x = (v1.twist.angular.x - v0.twist.angular.x) / dtv;
            acc_msg.accel.angular.y = (v1.twist.angular.y - v0.twist.angular.y) / dtv;
            acc_msg.accel.angular.z = (v1.twist.angular.z - v0.twist.angular.z) / dtv;
            computed_fut_acc = true;
          } else {
            RCLCPP_WARN(get_logger(), "futureDesiredPoseCallback: dtv too small or non-positive, skipping future desired acceleration publish");
          }
        } else {
          // size == 3: central difference
          const auto &v0 = last_future_desired_vel_buffer_[0];
          const auto &v2 = last_future_desired_vel_buffer_[2];
          rclcpp::Time t0(v0.header.stamp);
          rclcpp::Time t2(v2.header.stamp);
          double dtv_total = 0.002;
          if (dtv_total > 1e-6) {
            acc_msg.accel.linear.x  = 0;
            acc_msg.accel.linear.y  = 0;
            acc_msg.accel.linear.z  = 0;
            acc_msg.accel.angular.x = 0;
            acc_msg.accel.angular.y = 0;
            acc_msg.accel.angular.z = 0;

            computed_fut_acc = true;
          } else {
            RCLCPP_WARN(get_logger(), "futureDesiredPoseCallback: dtv_total too small, skipping future desired acceleration publish");
          }
        }
      }
      if (computed_fut_acc) {
        future_desired_ee_acceleration_pub_->publish(acc_msg);
        std::vector<float> seq_future_desired_acc = {
          static_cast<float>(acc_msg.accel.linear.x),
          static_cast<float>(acc_msg.accel.linear.y),
          static_cast<float>(acc_msg.accel.linear.z),
        };

        future_desired_acc_buffer.push_back(seq_future_desired_acc);

        if (future_desired_acc_buffer.size() > seq_length_) future_desired_acc_buffer.pop_front();

        if (future_desired_acc_buffer.size() == seq_length_)
        {
          std_msgs::msg::Float32MultiArray seq_future_desired_acc_msg;

          seq_future_desired_acc_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
          seq_future_desired_acc_msg.layout.dim[0].label = "sequence";
          seq_future_desired_acc_msg.layout.dim[0].size = seq_length_*3;

          // Flatten and copy
          for (const auto & sample: future_desired_acc_buffer)
          seq_future_desired_acc_msg.data.insert(seq_future_desired_acc_msg.data.end(), sample.begin(), sample.end());

          seq_future_desired_ee_acceleration_pub_->publish(seq_future_desired_acc_msg);
        }
      }
    }

    // After computing based on previous future-desired pose, update flag
    have_last_desired_ee_velocity_ = true;
  }


  // Desired pose callback: IK
  void desiredPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    // get the published message and fix the angles to RPY
    double x=msg->pose.position.x, y=msg->pose.position.y, z=msg->pose.position.z;
    double roll,pitch,yaw;
    quaternionToRPY(msg->pose.orientation.x,msg->pose.orientation.y,
                    msg->pose.orientation.z,msg->pose.orientation.w,
                    roll,pitch,yaw);

    // KDL::Frame takes rotation first for some reason
    // target = desired ee pose built from the incoming message
    KDL::Frame target(KDL::Rotation::RPY(roll,pitch,yaw), KDL::Vector(x,y,z));
    KDL::JntArray q0(kdl_chain_.getNrOfJoints());

    // fill in q0 as current_q_ (instead of an initial guess)
    // desired_q_ is where the results are stored = the joint positions that gets the ee to target

    for (size_t i = 0; i < q0.rows(); ++i) 
    {
      q0(i) = current_q_(i);
    }
    
    if (ik_solver_) {
      // inputs are q0 and target, to get the output desired_q_
      int status = ik_solver_->CartToJnt(q0, target, desired_q_);
      if (status < 0) {
        RCLCPP_WARN(get_logger(), "IK solver failed");
      }
    }

    // store desired EE pose for impedance 
    desired_pos_.x() = x;
    desired_pos_.y() = y;
    desired_pos_.z() = z;
    desired_rpy_.x() = roll;
    desired_rpy_.y() = pitch;
    desired_rpy_.z() = yaw;

    std::vector<float> seq_desired_pose = {
      static_cast<float>(msg->pose.position.x),
      static_cast<float>(msg->pose.position.y),
      static_cast<float>(msg->pose.position.z),
    };

    desired_pose_buffer.push_back(seq_desired_pose);

    if (desired_pose_buffer.size() > seq_length_) desired_pose_buffer.pop_front();

    if (desired_pose_buffer.size() == seq_length_)
    {
      std_msgs::msg::Float32MultiArray seq_desired_pose_msg;

      seq_desired_pose_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
      seq_desired_pose_msg.layout.dim[0].label = "sequence";
      seq_desired_pose_msg.layout.dim[0].size = seq_length_*3;

      // Flatten and copy
      for (const auto & sample: desired_pose_buffer)
      seq_desired_pose_msg.data.insert(seq_desired_pose_msg.data.end(), sample.begin(), sample.end());

      seq_desired_ee_pose_pub_->publish(seq_desired_pose_msg);
    }


    // publish desired EE pose 
    {
      geometry_msgs::msg::PoseStamped desired_msg;
      // copy header from incoming msg, so frame_id and stamp are consistent
      desired_msg.header = msg->header;
      desired_msg.pose = msg->pose;
      desired_ee_pose_pub_->publish(desired_msg);
    }
    have_optitrack_pose_ = true;

    // === ADDITION: compute and publish desired EE velocity ===
    // Buffer desired poses for velocity computation
    last_desired_pose_buffer_.push_back(*msg);
    if (last_desired_pose_buffer_.size() > 3) {
      last_desired_pose_buffer_.pop_front();
    }

    bool computed_des_vel = false;
    geometry_msgs::msg::TwistStamped des_vel_msg;
    des_vel_msg.header = msg->header;

    if (last_desired_pose_buffer_.size() >= 2) {
      if (last_desired_pose_buffer_.size() == 2) {
        // Forward difference between 0 and 1
        const auto &p0 = last_desired_pose_buffer_[0];
        const auto &p1 = last_desired_pose_buffer_[1];
        rclcpp::Time t0(p0.header.stamp);
        rclcpp::Time t1(p1.header.stamp);
        double dt = 0.001;
        if (dt > 1e-6) {
          // Linear velocity
          des_vel_msg.twist.linear.x = (p1.pose.position.x - p0.pose.position.x)/dt;
          des_vel_msg.twist.linear.y = (p1.pose.position.y - p0.pose.position.y)/dt;
          des_vel_msg.twist.linear.z = (p1.pose.position.z - p0.pose.position.z)/dt;
          // Angular velocity: quaternion difference over dt
          Eigen::Quaterniond q_prev(
            p0.pose.orientation.w,
            p0.pose.orientation.x,
            p0.pose.orientation.y,
            p0.pose.orientation.z
          );
          Eigen::Quaterniond q_cur(
            p1.pose.orientation.w,
            p1.pose.orientation.x,
            p1.pose.orientation.y,
            p1.pose.orientation.z
          );
          Eigen::Quaterniond q_delta = q_cur * q_prev.inverse();
          q_delta.normalize();
          double angle = 2.0 * std::acos(std::clamp(q_delta.w(), -1.0, 1.0));
          Eigen::Vector3d axis;
          double s = std::sqrt(1.0 - q_delta.w()*q_delta.w());
          if (s < 1e-6) {
            axis.setZero();
          } else {
            axis = Eigen::Vector3d(q_delta.x(), q_delta.y(), q_delta.z()) / s;
          }
          Eigen::Vector3d ang_vel = axis * (angle / dt);
          des_vel_msg.twist.angular.x = ang_vel.x();
          des_vel_msg.twist.angular.y = ang_vel.y();
          des_vel_msg.twist.angular.z = ang_vel.z();
          computed_des_vel = true;
        } else {
          RCLCPP_WARN(get_logger(), "desiredPoseCallback: dt too small or non-positive, skipping desired velocity publish");
        }
      } else {
        // size == 3: central difference between 0 and 2
        const auto &p0 = last_desired_pose_buffer_[0];
        const auto &p2 = last_desired_pose_buffer_[2];
        rclcpp::Time t0(p0.header.stamp);
        rclcpp::Time t2(p2.header.stamp);
        double dt_total = 0.002;
        if (dt_total > 1e-6) {
          // Linear velocity
          des_vel_msg.twist.linear.x = (p2.pose.position.x - p0.pose.position.x)/dt_total;
          des_vel_msg.twist.linear.y = (p2.pose.position.y - p0.pose.position.y)/dt_total;
          des_vel_msg.twist.linear.z = (p2.pose.position.z - p0.pose.position.z)/dt_total;
          // Angular velocity: quaternion difference over dt_total
          Eigen::Quaterniond q_prev(
            p0.pose.orientation.w,
            p0.pose.orientation.x,
            p0.pose.orientation.y,
            p0.pose.orientation.z
          );
          Eigen::Quaterniond q_cur(
            p2.pose.orientation.w,
            p2.pose.orientation.x,
            p2.pose.orientation.y,
            p2.pose.orientation.z
          );
          Eigen::Quaterniond q_delta = q_cur * q_prev.inverse();
          q_delta.normalize();
          double angle = 2.0 * std::acos(std::clamp(q_delta.w(), -1.0, 1.0));
          Eigen::Vector3d axis;
          double s = std::sqrt(1.0 - q_delta.w()*q_delta.w());
          if (s < 1e-6) {
            axis.setZero();
          } else {
            axis = Eigen::Vector3d(q_delta.x(), q_delta.y(), q_delta.z()) / s;
          }
          Eigen::Vector3d ang_vel = axis * (angle / dt_total);
          des_vel_msg.twist.angular.x = ang_vel.x();
          des_vel_msg.twist.angular.y = ang_vel.y();
          des_vel_msg.twist.angular.z = ang_vel.z();
          computed_des_vel = true;
        } else {
          RCLCPP_WARN(get_logger(), "desiredPoseCallback: total dt too small, skipping desired velocity publish");
        }
      }
    }

    if (computed_des_vel) {
      desired_ee_velocity_pub_->publish(des_vel_msg);

      std::vector<float> seq_desired_vel = {
          static_cast<float>(des_vel_msg.twist.linear.x),
          static_cast<float>(des_vel_msg.twist.linear.y),
          static_cast<float>(des_vel_msg.twist.linear.z),
        };

        desired_vel_buffer.push_back(seq_desired_vel);

        if (desired_vel_buffer.size() > seq_length_) desired_vel_buffer.pop_front();

        if (desired_vel_buffer.size() == seq_length_)
        {
          std_msgs::msg::Float32MultiArray seq_desired_vel_msg;

          seq_desired_vel_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
          seq_desired_vel_msg.layout.dim[0].label = "sequence";
          seq_desired_vel_msg.layout.dim[0].size = seq_length_*3;

          // Flatten and copy
          for (const auto & sample: desired_vel_buffer)
          seq_desired_vel_msg.data.insert(seq_desired_vel_msg.data.end(), sample.begin(), sample.end());

          seq_desired_ee_velocity_pub_->publish(seq_desired_vel_msg);
        }



      // Buffer this desired velocity for possible future desired acceleration (if needed)
      last_desired_vel_buffer_.push_back(des_vel_msg);
      if (last_desired_vel_buffer_.size() > 3) {
        last_desired_vel_buffer_.pop_front();
      }
      // === ADDITION: compute and publish  desired EE acceleration ===
      bool computed_des_acc = false;
      geometry_msgs::msg::AccelStamped acc_msg;
      acc_msg.header = des_vel_msg.header;
      if (last_desired_vel_buffer_.size() >= 2) {
        if (last_desired_vel_buffer_.size() == 2) {
          // Forward difference
          const auto &v0 = last_desired_vel_buffer_[0];
          const auto &v1 = last_desired_vel_buffer_[1];
          rclcpp::Time t0(v0.header.stamp);
          rclcpp::Time t1(v1.header.stamp);
          double dtv = 0.001;
          if (dtv > 1e-6) {
            acc_msg.accel.linear.x  = (v1.twist.linear.x  - v0.twist.linear.x ) / dtv;
            acc_msg.accel.linear.y  = (v1.twist.linear.y  - v0.twist.linear.y ) / dtv;
            acc_msg.accel.linear.z  = (v1.twist.linear.z  - v0.twist.linear.z ) / dtv;
            acc_msg.accel.angular.x = (v1.twist.angular.x - v0.twist.angular.x) / dtv;
            acc_msg.accel.angular.y = (v1.twist.angular.y - v0.twist.angular.y) / dtv;
            acc_msg.accel.angular.z = (v1.twist.angular.z - v0.twist.angular.z) / dtv;
            computed_des_acc = true;
          } else {
            RCLCPP_WARN(get_logger(), "futureDesiredPoseCallback: dtv too small or non-positive, skipping future desired acceleration publish");
          }
        } else {
          // size == 3: central difference
          const auto &v0 = last_desired_vel_buffer_[0];
          const auto &v2 = last_desired_vel_buffer_[2];
          rclcpp::Time t0(v0.header.stamp);
          rclcpp::Time t2(v2.header.stamp);
          double dtv_total = 0.001;
          if (dtv_total > 1e-6) {
            // acc_msg.accel.linear.x  = (v2.twist.linear.x  - v0.twist.linear.x ) / dtv_total;
            // acc_msg.accel.linear.y  = (v2.twist.linear.y  - v0.twist.linear.y ) / dtv_total;
            // acc_msg.accel.linear.z  = (v2.twist.linear.z  - v0.twist.linear.z ) / dtv_total;
            // acc_msg.accel.angular.x = (v2.twist.angular.x - v0.twist.angular.x) / dtv_total;
            // acc_msg.accel.angular.y = (v2.twist.angular.y - v0.twist.angular.y) / dtv_total;
            // acc_msg.accel.angular.z = (v2.twist.angular.z - v0.twist.angular.z) / dtv_total;

            acc_msg.accel.linear.x  = 0;
            acc_msg.accel.linear.y  = 0;
            acc_msg.accel.linear.z  = 0;
            acc_msg.accel.angular.x = 0;
            acc_msg.accel.angular.y = 0;
            acc_msg.accel.angular.z = 0;

            computed_des_acc = true;
          } else {
            RCLCPP_WARN(get_logger(), "futureDesiredPoseCallback: dtv_total too small, skipping future desired acceleration publish");
          }
        }
      }
      if (computed_des_acc) {
        desired_ee_acceleration_pub_->publish(acc_msg);
        
        std::vector<float> seq_desired_acc = {
          static_cast<float>(0),
          static_cast<float>(0),
          static_cast<float>(0),
        };

        desired_acc_buffer.push_back(seq_desired_acc);

        if (desired_acc_buffer.size() > seq_length_) desired_acc_buffer.pop_front();

        if (desired_acc_buffer.size() == seq_length_)
        {
          std_msgs::msg::Float32MultiArray seq_desired_acc_msg;

          seq_desired_acc_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
          seq_desired_acc_msg.layout.dim[0].label = "sequence";
          seq_desired_acc_msg.layout.dim[0].size = seq_length_*3;

          // Flatten and copy
          for (const auto & sample: desired_acc_buffer)
          seq_desired_acc_msg.data.insert(seq_desired_acc_msg.data.end(), sample.begin(), sample.end());

          seq_desired_ee_acceleration_pub_->publish(seq_desired_acc_msg);
        }
      }
      have_last_desired_ee_velocity_ = true;
    }
    // ==========================================================

  }

  // EE wrench callback: store latest raw wrench
  void eeWrenchCallback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
    latest_raw_wrench_(0) = msg->wrench.force.x;
    latest_raw_wrench_(1) = msg->wrench.force.y;
    latest_raw_wrench_(2) = msg->wrench.force.z;
    latest_raw_wrench_(3) = msg->wrench.torque.x;
    latest_raw_wrench_(4) = msg->wrench.torque.y;
    latest_raw_wrench_(5) = msg->wrench.torque.z;
    have_raw_wrench_ = true;
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    auto now_time = now();
    size_t nj = current_q_.rows();

    // check if the message size is correct
    if (msg->position.size() != nj) return;
    
    // fill in current joint angles and velocity
    for (size_t i=0; i<nj; ++i) {
      current_q_(i)  = msg->position[i];
      current_dq_(i) = msg->velocity[i];
    }

    updateAcceleration(now_time);

    // --- Compute and publish actual EE pose via FK ---
    if (fk_solver_) {
      KDL::Frame actual_frame;
      if (fk_solver_->JntToCart(current_q_, actual_frame) >= 0) {
        // Convert KDL::Frame to PoseStamped
        geometry_msgs::msg::PoseStamped actual_msg;
        actual_msg.header = msg->header;
        actual_msg.pose.position.x = actual_frame.p.x();
        actual_msg.pose.position.y = actual_frame.p.y();
        actual_msg.pose.position.z = actual_frame.p.z();
        double qx, qy, qz, qw;
        actual_frame.M.GetQuaternion(qx, qy, qz, qw);
        actual_msg.pose.orientation.x = qx;
        actual_msg.pose.orientation.y = qy;
        actual_msg.pose.orientation.z = qz;
        actual_msg.pose.orientation.w = qw;
        actual_ee_pose_pub_->publish(actual_msg);

        // store actual EE pose for impedance 
        actual_pos_.x() = actual_frame.p.x();
        actual_pos_.y() = actual_frame.p.y();
        actual_pos_.z() = actual_frame.p.z();
        double roll_act, pitch_act, yaw_act;
        quaternionToRPY(qx, qy, qz, qw, roll_act, pitch_act, yaw_act);
        actual_rpy_.x() = roll_act;
        actual_rpy_.y() = pitch_act;
        actual_rpy_.z() = yaw_act;

        std::vector<float> seq_actual_pose = {
          static_cast<float>(actual_msg.pose.position.x),
          static_cast<float>(actual_msg.pose.position.y),
          static_cast<float>(actual_msg.pose.position.z),
        };

        actual_pose_buffer.push_back(seq_actual_pose);

        if (actual_pose_buffer.size() > seq_length_) actual_pose_buffer.pop_front();

        if (actual_pose_buffer.size() == seq_length_)
        {
          std_msgs::msg::Float32MultiArray seq_actual_pose_msg;

          seq_actual_pose_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
          seq_actual_pose_msg.layout.dim[0].label = "sequence";
          seq_actual_pose_msg.layout.dim[0].size = seq_length_*3;

          // Flatten and copy
          for (const auto & sample: actual_pose_buffer)
          seq_actual_pose_msg.data.insert(seq_actual_pose_msg.data.end(), sample.begin(), sample.end());

          seq_actual_ee_pose_pub_->publish(seq_actual_pose_msg);
        }



        // --- Compute and publish actual EE velocity via Jacobian * qdot ---
        if (jac_solver_) {
          // Compute Jacobian at current_q_
          if (jac_solver_->JntToJac(current_q_, jacobian_) >= 0) {
            // Build Eigen J matrix 6x7
            Eigen::Matrix<double,6,7> J;
            for (int r=0; r<6; ++r) {
              for (int c=0; c<7; ++c) {
                J(r,c) = jacobian_(r,c);
              }
            }

            // Build joint velocity vector
            Eigen::Matrix<double,7,1> qdot_eig;
            for (int i=0; i<7; ++i) {
              qdot_eig(i) = current_dq_(i);
            }
            // Compute Cartesian twist: [v; omega] = J * qdot --> EE velocity
            Eigen::Matrix<double,6,1> twist = J * qdot_eig;

            // Apply EWMA filtering to actual EE velocity before publishing
            Eigen::Matrix<double,6,1> current_vel = twist;
            Eigen::Matrix<double,6,1> filtered_vel;
            if (first_actual_vel_) {
              filtered_actual_ee_vel_ = current_vel;
              first_actual_vel_ = false;
            } else {
              filtered_actual_ee_vel_ = ee_vel_ewma_alpha_ * filtered_actual_ee_vel_ 
                                      + (1.0 - ee_vel_ewma_alpha_) * current_vel;
            }
            filtered_vel = filtered_actual_ee_vel_;

            // Publish as TwistStamped
            geometry_msgs::msg::TwistStamped vel_msg;
            vel_msg.header = msg->header;
            // linear
            vel_msg.twist.linear.x  = filtered_vel(0);
            vel_msg.twist.linear.y  = filtered_vel(1);
            vel_msg.twist.linear.z  = filtered_vel(2);
            // angular
            vel_msg.twist.angular.x = filtered_vel(3);
            vel_msg.twist.angular.y = filtered_vel(4);
            vel_msg.twist.angular.z = filtered_vel(5);
            actual_ee_velocity_pub_->publish(vel_msg);

            std::vector<float> seq_actual_vel = {
              static_cast<float>(filtered_vel(0)),
              static_cast<float>(filtered_vel(1)),
              static_cast<float>(filtered_vel(2)),
            };

            actual_vel_buffer.push_back(seq_actual_vel);

            if (actual_vel_buffer.size() > seq_length_) actual_vel_buffer.pop_front();

            if (actual_vel_buffer.size() == seq_length_)
            {
              std_msgs::msg::Float32MultiArray seq_actual_vel_msg;

              seq_actual_vel_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
              seq_actual_vel_msg.layout.dim[0].label = "sequence";
              seq_actual_vel_msg.layout.dim[0].size = seq_length_*3;

              // Flatten and copy
              for (const auto & sample: actual_vel_buffer)
              seq_actual_vel_msg.data.insert(seq_actual_vel_msg.data.end(), sample.begin(), sample.end());

              seq_actual_ee_velocity_pub_->publish(seq_actual_vel_msg);
            }



            // === ADDITION: compute and publish actual EE acceleration via buffered finite differences and filter ===
            // Buffer this filtered actual velocity
            last_actual_ee_vel_buffer_.push_back(vel_msg);
            if (last_actual_ee_vel_buffer_.size() > 3) last_actual_ee_vel_buffer_.pop_front();

            if (last_actual_ee_vel_buffer_.size() >= 2) {
              geometry_msgs::msg::AccelStamped acc_msg;
              acc_msg.header = vel_msg.header;
              bool computed_acc = false;
              Eigen::Matrix<double,6,1> raw_acc;
              if (last_actual_ee_vel_buffer_.size() == 2) {
                // forward diff
                const auto &v0 = last_actual_ee_vel_buffer_[0];
                const auto &v1 = last_actual_ee_vel_buffer_[1];
                rclcpp::Time t0(v0.header.stamp);
                rclcpp::Time t1(v1.header.stamp);
                double dtv = 0.001;
                if (dtv > 1e-6) {
                  raw_acc(0) = (v1.twist.linear.x  - v0.twist.linear.x ) / dtv;
                  raw_acc(1) = (v1.twist.linear.y  - v0.twist.linear.y ) / dtv;
                  raw_acc(2) = (v1.twist.linear.z  - v0.twist.linear.z ) / dtv;
                  raw_acc(3) = (v1.twist.angular.x - v0.twist.angular.x) / dtv;
                  raw_acc(4) = (v1.twist.angular.y - v0.twist.angular.y) / dtv;
                  raw_acc(5) = (v1.twist.angular.z - v0.twist.angular.z) / dtv;
                  computed_acc = true;
                } else {
                  RCLCPP_WARN(get_logger(), "jointStateCallback: dt too small or non-positive, skipping actual EE acceleration publish");
                }
              } else {
                // size==3: central diff
                const auto &v0 = last_actual_ee_vel_buffer_[0];
                const auto &v2 = last_actual_ee_vel_buffer_[2];
                rclcpp::Time t0(v0.header.stamp);
                rclcpp::Time t2(v2.header.stamp);
                double dtv_total = 0.002;
                if (dtv_total > 1e-6) {
                  raw_acc(0) = (v2.twist.linear.x  - v0.twist.linear.x ) / dtv_total;
                  raw_acc(1) = (v2.twist.linear.y  - v0.twist.linear.y ) / dtv_total;
                  raw_acc(2) = (v2.twist.linear.z  - v0.twist.linear.z ) / dtv_total;
                  raw_acc(3) = (v2.twist.angular.x - v0.twist.angular.x) / dtv_total;
                  raw_acc(4) = (v2.twist.angular.y - v0.twist.angular.y) / dtv_total;
                  raw_acc(5) = (v2.twist.angular.z - v0.twist.angular.z) / dtv_total;
                  computed_acc = true;
                } else {
                  RCLCPP_WARN(get_logger(), "jointStateCallback: total dt too small, skipping actual EE acceleration publish");
                }
              }
              if (computed_acc) {
                // Filter acceleration via EWMA before publishing
                if (first_actual_acc_) {
                  filtered_actual_ee_accel_ = raw_acc;
                  first_actual_acc_ = false;
                } else {
                  filtered_actual_ee_accel_ = ee_accel_ewma_alpha_ * filtered_actual_ee_accel_
                                            + (1.0 - ee_accel_ewma_alpha_) * raw_acc;
                }
                // Fill acc_msg from filtered_actual_ee_accel_
                acc_msg.accel.linear.x  = filtered_actual_ee_accel_(0);
                acc_msg.accel.linear.y  = filtered_actual_ee_accel_(1);
                acc_msg.accel.linear.z  = filtered_actual_ee_accel_(2);
                acc_msg.accel.angular.x = filtered_actual_ee_accel_(3);
                acc_msg.accel.angular.y = filtered_actual_ee_accel_(4);
                acc_msg.accel.angular.z = filtered_actual_ee_accel_(5);
                actual_ee_acceleration_pub_->publish(acc_msg);
              }

              std::vector<float> seq_actual_acc = {
                static_cast<float>(filtered_actual_ee_accel_(0)),
                static_cast<float>(filtered_actual_ee_accel_(1)),
                static_cast<float>(filtered_actual_ee_accel_(2)),
              };

              actual_acc_buffer.push_back(seq_actual_acc);

              if (actual_acc_buffer.size() > seq_length_) actual_acc_buffer.pop_front();

              if (actual_acc_buffer.size() == seq_length_)
              {
                std_msgs::msg::Float32MultiArray seq_actual_acc_msg;

                seq_actual_acc_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
                seq_actual_acc_msg.layout.dim[0].label = "sequence";
                seq_actual_acc_msg.layout.dim[0].size = seq_length_*3;

                // Flatten and copy
                for (const auto & sample: actual_acc_buffer)
                seq_actual_acc_msg.data.insert(seq_actual_acc_msg.data.end(), sample.begin(), sample.end());

                seq_actual_ee_acceleration_pub_->publish(seq_actual_acc_msg);
              }
            }
            // Update last actual EE velocity
            last_actual_ee_velocity_msg_ = vel_msg;
            have_last_actual_ee_velocity_ = true;

            if (!have_optitrack_pose_) {
              geometry_msgs::msg::PoseStamped desired_msg = actual_msg;

              // keep same header/frame
              // Override position with initial fixed desired pose until a new message arrives
              desired_msg.pose.position.x = 0.6;
              desired_msg.pose.position.y = 0.0;
              desired_msg.pose.position.z = 0.41;

              // set orientation to roll=0.0, pitch=pi, yaw=0.0
              {
                double roll_init = 0.0;
                double pitch_init = M_PI;
                double yaw_init = 0.0;
                double cy = std::cos(yaw_init * 0.5);
                double sy = std::sin(yaw_init * 0.5);
                double cp = std::cos(pitch_init * 0.5);
                double sp = std::sin(pitch_init * 0.5);
                double cr = std::cos(roll_init * 0.5);
                double sr = std::sin(roll_init * 0.5);

                double qw_init = cr * cp * cy + sr * sp * sy;
                double qx_init = sr * cp * cy - cr * sp * sy;
                double qy_init = cr * sp * cy + sr * cp * sy;
                double qz_init = cr * cp * sy - sr * sp * cy;

                desired_msg.pose.orientation.x = qx_init;
                desired_msg.pose.orientation.y = qy_init;
                desired_msg.pose.orientation.z = qz_init;
                desired_msg.pose.orientation.w = qw_init;

                // make controller internal desired pose consistent with the chosen initial fixed pose
                desired_pos_.x() = 0.6;
                desired_pos_.y() = 0.0;
                desired_pos_.z() = 0.41;
                desired_rpy_.x() = roll_init;
                desired_rpy_.y() = pitch_init;
                desired_rpy_.z() = yaw_init;
              }

              desired_ee_pose_pub_->publish(desired_msg);

              // publish actual EE velocity as desired velocity as well
              // desired_ee_velocity_pub_->publish(vel_msg);

              // buffer as if it were last desired (so later computations don't break)
              last_desired_pose_buffer_.push_back(desired_msg);
              if (last_desired_pose_buffer_.size() > 3) last_desired_pose_buffer_.pop_front();

              // For velocity buffer, use zero velocity so the controller holds the fixed initial pose
              geometry_msgs::msg::TwistStamped zero_vel_msg;
              zero_vel_msg.header = vel_msg.header;
              zero_vel_msg.twist.linear.x  = 0.0;
              zero_vel_msg.twist.linear.y  = 0.0;
              zero_vel_msg.twist.linear.z  = 0.0;
              zero_vel_msg.twist.angular.x = 0.0;
              zero_vel_msg.twist.angular.y = 0.0;
              zero_vel_msg.twist.angular.z = 0.0;

              last_desired_vel_buffer_.push_back(zero_vel_msg);
              if (last_desired_vel_buffer_.size() > 3) last_desired_vel_buffer_.pop_front();

              // tell the code we have a desired velocity available (consistent behavior)
              have_last_desired_ee_velocity_ = true;
            }

            // --- NEW: same for future desired pose/velocity if no future optitrack ---
            if (!have_future_optitrack_pose_) {
              geometry_msgs::msg::PoseStamped future_msg = actual_msg;

              // Override position with initial fixed future desired pose until a new future message arrives
              future_msg.pose.position.x = 0.6;
              future_msg.pose.position.y = 0.0;
              future_msg.pose.position.z = 0.41;

              // set future orientation to roll=0.0, pitch=pi, yaw=0.0
              {
                double roll_f = 0.0;
                double pitch_f = M_PI;
                double yaw_f = 0.0;
                double cy = std::cos(yaw_f * 0.5);
                double sy = std::sin(yaw_f * 0.5);
                double cp = std::cos(pitch_f * 0.5);
                double sp = std::sin(pitch_f * 0.5);
                double cr = std::cos(roll_f * 0.5);
                double sr = std::sin(roll_f * 0.5);

                double qw_f = cr * cp * cy + sr * sp * sy;
                double qx_f = sr * cp * cy - cr * sp * sy;
                double qy_f = cr * sp * cy + sr * cp * sy;
                double qz_f = cr * cp * sy - sr * sp * cy;

                future_msg.pose.orientation.x = qx_f;
                future_msg.pose.orientation.y = qy_f;
                future_msg.pose.orientation.z = qz_f;
                future_msg.pose.orientation.w = qw_f;
              }

              future_desired_ee_pose_pub_->publish(future_msg);

              // For future desired velocity, use zero so controller will hold the fixed future pose
              geometry_msgs::msg::TwistStamped zero_fut_vel_msg;
              zero_fut_vel_msg.header = vel_msg.header;
              zero_fut_vel_msg.twist.linear.x  = 0.0;
              zero_fut_vel_msg.twist.linear.y  = 0.0;
              zero_fut_vel_msg.twist.linear.z  = 0.0;
              zero_fut_vel_msg.twist.angular.x = 0.0;
              zero_fut_vel_msg.twist.angular.y = 0.0;
              zero_fut_vel_msg.twist.angular.z = 0.0;
              future_desired_ee_velocity_pub_->publish(zero_fut_vel_msg);

              last_future_desired_pose_buffer_.push_back(future_msg);
              if (last_future_desired_pose_buffer_.size() > 3) last_future_desired_pose_buffer_.pop_front();

              last_future_desired_vel_buffer_.push_back(zero_fut_vel_msg);
              if (last_future_desired_vel_buffer_.size() > 3) last_future_desired_vel_buffer_.pop_front();
            }
            // =========================================================

          } else {
            RCLCPP_WARN(get_logger(), "KDL Jacobian computation failed for velocity");
          }
        }
      } else {
        RCLCPP_WARN(get_logger(), "FK solver failed to compute actual EE pose");
      }
    }
    // ----------------------------------------------------

    // ----------------------------------------------------

    // Compute impedance reference position and velocity

    Eigen::Matrix<double,6,1> F_filt  = Eigen::Matrix<double,6,1>::Zero();
    bool have_impedance = false;
   

    // -------- Force estimation via EE wrench mapping with tau_bias and gravity subtraction --------

    // check if xml, mujoco model, and sensors are initialized correctly 
    if (have_raw_wrench_ && mj_model_ && mj_data_) {

      // 1) Compute Jacobian J(q) 6x7

      for (int i = 0; i < (int)nj; ++i) q_kdl_(i) = current_q_(i);

      // check if the jacobian is calculated correctly 
      if (jac_solver_->JntToJac(q_kdl_, jacobian_) < 0) {

        RCLCPP_WARN(get_logger(), "KDL Jacobian computation failed");
        // Even if Jacobian fails, we skip impedance but continue PID below.
      } else {
        // fill the jacobian_ data into J
        Eigen::Matrix<double,6,7> J;
        for (int r=0;r<6;++r) for (int c=0;c<7;++c) J(r,c) = jacobian_(r,c);

        // 2) Raw EE wrench from MuJoCo; latest_raw_wrench_ gathered in the eeWrenchCallback --> fill it in F_raw
        Eigen::Matrix<double,6,1> F_raw = latest_raw_wrench_;

        // 3) Joint-space external torque from raw wrench
        Eigen::Matrix<double,7,1> tau_ext_raw = J.transpose() * F_raw;

        // 4) Compute tau_bias via MuJoCo inverse dynamics
        //    Set MuJoCo qpos, qvel
        for (int i = 0; i < (int)nj; ++i) {
          mj_data_->qpos[i] = current_q_(i);
          mj_data_->qvel[i] = current_dq_(i);
          // mj_data_->qacc not needed for mj_inverse
        }
        
        // Mujoco provides tau_bias from qfrc_bias which includes Coriolis, centrifugal, and gravitational
        // Gravitational forces already compensated by the controller in tau_command so we need to remove it from tau_bias

        mj_inverse(mj_model_, mj_data_);
        Eigen::Matrix<double,7,1> tau_bias;
        for (int i = 0; i < (int)nj; ++i) {
          tau_bias(i) = mj_data_->qfrc_bias[i];
        }

        // Mujoco doesn't provide gravitational force alone, so we go to KDL to get tau_g
        dyn_param_->JntToGravity(current_q_, tau_g_);

        // 5) Subtract bias
        // ---> tau_bias - tau_g = external force related to Coriolis and centrifugal forces
        Eigen::Matrix<double,7,1> tau_ext_corrected = tau_ext_raw
          - Eigen::Map<Eigen::Matrix<double,7,1>>(tau_g_.data.data())
          + tau_bias;

        // 6) Filter in joint space (EWMA). Use alpha close to 1
        if (first_tau_ext_) {
          filtered_tau_ext_ = tau_ext_corrected;
          first_tau_ext_ = false;
        } else {
          filtered_tau_ext_ = ewma_alpha_ * filtered_tau_ext_ 
                            + (1.0 - ewma_alpha_) * tau_ext_corrected;
        }

        // 7) Map back to Cartesian cleaned wrench if needed:

        Eigen::Matrix<double,6,6> JJt = J * J.transpose();
        for (int i=0;i<6;++i) JJt(i,i) += damp_lambda_*damp_lambda_;

        // calculate filtered cartesian forces
        Eigen::Matrix<double,6,1> J_tau_filtered = J * filtered_tau_ext_;
        F_filt = JJt.ldlt().solve(J_tau_filtered);

        // start the impedance controller once you fill in F_filt
        have_impedance = true;
      }

      // 9) Publish cleaned EE wrench
      geometry_msgs::msg::WrenchStamped filt_msg;
      filt_msg.header = msg->header;
      filt_msg.wrench.force.x  = F_filt(0);
      filt_msg.wrench.force.y  = F_filt(1);
      filt_msg.wrench.force.z  = F_filt(2);
      filt_msg.wrench.torque.x = F_filt(3);
      filt_msg.wrench.torque.y = F_filt(4);
      filt_msg.wrench.torque.z = F_filt(5);
      ee_contact_wrench_filtered_pub_->publish(filt_msg);

      std::vector<float> seq_wrench = {
        static_cast<float>(F_filt(0)),
        static_cast<float>(F_filt(1)),
        static_cast<float>(F_filt(2)),
      };

      wrench_buffer.push_back(seq_wrench);

      if (wrench_buffer.size() > seq_length_) wrench_buffer.pop_front();

      if (wrench_buffer.size() == seq_length_)
      {
        std_msgs::msg::Float32MultiArray seq_wrench_msg;

        seq_wrench_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
        seq_wrench_msg.layout.dim[0].label = "sequence";
        seq_wrench_msg.layout.dim[0].size = seq_length_*3;

        // Flatten and copy
        for (const auto & sample: wrench_buffer)
        seq_wrench_msg.data.insert(seq_wrench_msg.data.end(), sample.begin(), sample.end());

        seq_ee_contact_wrench_filtered_pub_->publish(seq_wrench_msg);
      }


    }

    //  Impedance offset dynamics (anchored around moving desired) and compute impedance reference pose and velocity

    // Use KDL::JntArray for desired_q_imp:
    Eigen::Matrix<double,7,1> qdot_imp;  // unused directly, but placeholder

    KDL::JntArray desired_q_imp_(nj);
    KDL::JntArray desired_dq_imp_(nj);

    // Default fallback: desired_q_imp_ = desired_q_, desired_dq_imp_ = zero
    // for (size_t i = 0; i < nj; ++i) {
    //   desired_q_imp_(i) = desired_q_(i);
    //   desired_dq_imp_(i) = 0.0;
    // }

    if (have_impedance) {
      // 1) Update offset dynamics: M * ddot_offset + D * dot_offset + K * offset = F_filt
      // offset = imp_pos + current_pos - desired_pos
      Eigen::Matrix<double,6,1> acc_offset;
      // translational acceleration:
      acc_offset.head<3>() = M_inv_.block<3,3>(0,0) * (
                              F_filt.head<3>()
                              - K_cart_.block<3,3>(0,0) * offset_pos_.head<3>()
                              - D_cart_.block<3,3>(0,0) * offset_vel_.head<3>()
                            );
      // rotational acceleration:
      acc_offset.tail<3>() = M_inv_.block<3,3>(3,3) * (
                              F_filt.tail<3>()
                              - K_cart_.block<3,3>(3,3) * offset_pos_.tail<3>()
                              - D_cart_.block<3,3>(3,3) * offset_vel_.tail<3>()
                            );
      // double dt = 1/1000;
      double dt = last_dt_;
      offset_vel_ += acc_offset * dt;
      offset_pos_ += offset_vel_ * dt;

      Eigen::Vector3d imp_pos = desired_pos_ + offset_pos_.head<3>();
      Eigen::Vector3d imp_rpy = desired_rpy_  + offset_pos_.tail<3>();

      Eigen::Vector3d imp_vel_t =  offset_vel_.head<3>();
      Eigen::Vector3d imp_vel_r =  offset_vel_.tail<3>();

      // Build and publish PoseStamped:
      geometry_msgs::msg::PoseStamped imp_msg;
      imp_msg.header.stamp = msg->header.stamp;
      // frame_id: same as desired_ee_pose topic?
      imp_msg.header.frame_id = msg->header.frame_id;  
      imp_msg.pose.position.x = imp_pos.x();
      imp_msg.pose.position.y = imp_pos.y();
      imp_msg.pose.position.z = imp_pos.z();
      // Convert RPY back to quaternion:
      {
        double cr = std::cos(imp_rpy.x()*0.5), sr = std::sin(imp_rpy.x()*0.5);
        double cp = std::cos(imp_rpy.y()*0.5), sp = std::sin(imp_rpy.y()*0.5);
        double cy = std::cos(imp_rpy.z()*0.5), sy = std::sin(imp_rpy.z()*0.5);
        // quaternion in RPY order (roll, pitch, yaw):
        double qw = cr*cp*cy + sr*sp*sy;
        double qx = sr*cp*cy - cr*sp*sy;
        double qy = cr*sp*cy + sr*cp*sy;
        double qz = cr*cp*sy - sr*sp*cy;
        imp_msg.pose.orientation.x = qx;
        imp_msg.pose.orientation.y = qy;
        imp_msg.pose.orientation.z = qz;
        imp_msg.pose.orientation.w = qw;
      }

      impedance_ref_pose_pub_->publish(imp_msg);
      
      // 3) Compute IK for impedance reference pose to get desired_q_imp_
      if (ik_solver_) {
        KDL::Frame target_imp(
          KDL::Rotation::RPY(imp_rpy.x(), imp_rpy.y(), imp_rpy.z()),
          KDL::Vector(imp_pos.x(), imp_pos.y(), imp_pos.z())
        );
        KDL::JntArray q0(kdl_chain_.getNrOfJoints());
        for (size_t i = 0; i < q0.rows(); ++i) {
          q0(i) = current_q_(i);
        }
        int status = ik_solver_->CartToJnt(q0, target_imp, desired_q_imp_);
        if (status < 0) {
          RCLCPP_WARN(get_logger(), "Impedance IK solver failed; using previous desired_q_imp_");
          // desired_q_imp_ remains as previous desired_q_imp_
        }
      }

      // 4) Compute velocity IK for impedance reference twist to get desired_dq_imp_
      if (vel_ik_solver_) {
        // Build desired twist: linear + angular velocities.
        // Note: imp_vel_t is linear velocity (m/s), imp_vel_r is angular velocity (rad/s).
        KDL::Twist desired_twist(
          KDL::Vector(imp_vel_t.x(), imp_vel_t.y(), imp_vel_t.z()),
          KDL::Vector(imp_vel_r.x(), imp_vel_r.y(), imp_vel_r.z())
        );
        KDL::JntArray q0(kdl_chain_.getNrOfJoints());
        for (size_t i = 0; i < q0.rows(); ++i) {
          q0(i) = current_q_(i);
        }
        int vstatus = vel_ik_solver_->CartToJnt(q0, desired_twist, desired_dq_imp_);
        if (vstatus < 0) {
          RCLCPP_WARN(get_logger(), "Velocity IK solver failed; using previous desired_dq_imp_ - UUUUUUUUUUUUUU");
          // desired_dq_imp_ remains as previous or zero
        }
      }
      

    }

    // =======================================================

    // --- Now perform PID controller calculation and publication of joint_commands ---
    {
      Eigen::Matrix<double,7,1> e_q, e_dq;
      for (int i=0; i<7; ++i) {
        e_q(i)   = desired_q_imp_(i) - current_q_(i);
        e_dq(i)  = desired_dq_imp_(i) - current_dq_(i);
        integral_joint_error_(i) += e_q(i) * 0.001;
      }
      mj_inverse(mj_model_, mj_data_);
        Eigen::Matrix<double,7,1> tau_bias;
        for (int i = 0; i < (int)nj; ++i) {
          tau_bias(i) = mj_data_->qfrc_bias[i];
        }
      Eigen::Matrix<double,7,1> tau_pid = Kp_joint_.cwiseProduct(e_q)
                                       + Kd_joint_.cwiseProduct(e_dq)
                                       + Ki_joint_.cwiseProduct(integral_joint_error_);
      // First saturate rate based on last_torque_
      Eigen::Matrix<double,7,1> tau_cmd = saturateTorqueRate(tau_pid) - Eigen::Map<Eigen::Matrix<double,7,1>>(tau_g_.data.data())
          + tau_bias ;

      // --- Apply EWMA filtering to commanded torque ---
      if (first_cmd_) {
        filtered_cmd_ = tau_cmd;
        first_cmd_ = false;
      } else {
        filtered_cmd_ = cmd_ewma_alpha_ * filtered_cmd_ + (1.0 - cmd_ewma_alpha_) * tau_cmd;
      }
      // Update last_torque_ to the filtered output, so rate saturation uses filtered values next cycle
      last_torque_ = filtered_cmd_;

      // Publish filtered_cmd_
      std_msgs::msg::Float64MultiArray cmd; cmd.data.resize(nj);
      for (int i = 0; i < (int)nj; ++i) {
        cmd.data[i] = filtered_cmd_(i);
      }
      joint_command_pub_->publish(cmd);
    }

    // Note: the force estimation and impedance logic above runs every cycle if wrench is available.
  }

  // Members
  std::string urdf_path_, urdf_xml_, base_link_, tip_link_;
  KDL::Chain kdl_chain_;
  std::shared_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_;
  std::shared_ptr<KDL::ChainIkSolverPos_LMA> ik_solver_;
  std::shared_ptr<KDL::ChainIkSolverVel_pinv> vel_ik_solver_;    // <-- velocity IK solver
  std::shared_ptr<KDL::ChainDynParam> dyn_param_;

  KDL::JntArray current_q_, current_dq_, desired_q_, tau_g_;
  KDL::JntArray prev_dq_;
  Eigen::VectorXd qddot_;
  KDL::Jacobian jacobian_;
  KDL::JntArray q_kdl_;
  
  // PID
  Eigen::Matrix<double,7,1> Kp_joint_, Kd_joint_, Ki_joint_, integral_joint_error_, last_torque_;
  double delta_tau_max_;

  // Filtering & mapping
  double ewma_alpha_;
  double damp_lambda_;
  Eigen::Matrix<double,6,1> latest_raw_wrench_;
  bool have_raw_wrench_;
  Eigen::Matrix<double,7,1> filtered_tau_ext_;
  bool first_tau_ext_;

  // Command torque filtering
  double cmd_ewma_alpha_;
  Eigen::Matrix<double,7,1> filtered_cmd_;
  bool first_cmd_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tau_ext_pub_;

  // ROS subs/pubs
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr desired_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr ee_wrench_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_pub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr  ee_contact_wrench_filtered_pub_; //ee_contact_wrench_pub_,

  // === ADDITION: publishers for EE poses, velocities, accelerations, and impedance ref ===
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr desired_ee_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr actual_ee_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr actual_ee_velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr actual_ee_acceleration_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr impedance_ref_pose_pub_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_actual_ee_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_actual_ee_velocity_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_actual_ee_acceleration_pub_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_desired_ee_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_desired_ee_velocity_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_desired_ee_acceleration_pub_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_future_desired_ee_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_future_desired_ee_velocity_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_future_desired_ee_acceleration_pub_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr seq_ee_contact_wrench_filtered_pub_;
  
  

  // ==================================================

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr future_pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr future_desired_ee_pose_pub_;

  // === ADDITION: publisher for desired EE velocity ===
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr desired_ee_velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr desired_ee_acceleration_pub_;

  // ==================================================

  // === ADDITION: publishers for future desired EE velocity & acceleration ===
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr future_desired_ee_velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr future_desired_ee_acceleration_pub_;
  // ==================================================

  // For storing last desired pose to compute desired velocity
  geometry_msgs::msg::PoseStamped last_desired_pose_msg_;
  bool have_last_desired_pose_;

  // For storing last actual EE velocity to compute acceleration
  geometry_msgs::msg::TwistStamped last_actual_ee_velocity_msg_;
  bool have_last_actual_ee_velocity_;

  // For storing last desired EE velocity to compute future desired acceleration
  geometry_msgs::msg::TwistStamped last_desired_ee_velocity_msg_;
  bool have_last_desired_ee_velocity_;

  // Buffers for finite-difference:
  std::deque<std::pair<rclcpp::Time, KDL::JntArray>> dq_buffer_;
  std::deque<geometry_msgs::msg::TwistStamped> last_actual_ee_vel_buffer_;
  std::deque<geometry_msgs::msg::PoseStamped> last_desired_pose_buffer_;
  std::deque<geometry_msgs::msg::TwistStamped> last_desired_vel_buffer_;
  std::deque<geometry_msgs::msg::PoseStamped> last_future_desired_pose_buffer_;
  std::deque<geometry_msgs::msg::TwistStamped> last_future_desired_vel_buffer_;

  std::deque<std::vector<float>> wrench_buffer;

  std::deque<std::vector<float>> desired_pose_buffer;
  std::deque<std::vector<float>> desired_vel_buffer;
  std::deque<std::vector<float>> desired_acc_buffer;
  std::deque<std::vector<float>> actual_pose_buffer;
  std::deque<std::vector<float>> actual_vel_buffer;
  std::deque<std::vector<float>> actual_acc_buffer;
  std::deque<std::vector<float>> future_desired_pose_buffer;
  std::deque<std::vector<float>> future_desired_vel_buffer;
  std::deque<std::vector<float>> future_desired_acc_buffer;

  const size_t seq_length_;

  // Filtering state for actual EE velocity/acceleration
  Eigen::Matrix<double,6,1> filtered_actual_ee_vel_;
  Eigen::Matrix<double,6,1> filtered_actual_ee_accel_;
  bool first_actual_vel_;
  bool first_actual_acc_;
  double ee_vel_ewma_alpha_;
  double ee_accel_ewma_alpha_;

  // Time keeping
  rclcpp::Time prev_time_;
  double last_dt_;

  // MuJoCo members
  std::string mujoco_xml_path_;
  mjModel* mj_model_;
  mjData*  mj_data_;
  int      panda_wrist_body_id_;

  bool freeze_;
  KDL::JntArray frozen_q_;

  // === ADDITION: Impedance state and parameters ===
  Eigen::Matrix<double,6,1> offset_pos_;
  Eigen::Matrix<double,6,1> offset_vel_;
  double mass_imp_, inertia_imp_;
  double stiffness_trans_, stiffness_rot_;
  double damping_factor_trans_, damping_factor_rot_;
  Eigen::Matrix<double,6,6> M_inv_, K_cart_, D_cart_, M_cart_;
  // For storing poses
  Eigen::Vector3d desired_pos_;
  Eigen::Vector3d desired_rpy_;
  Eigen::Vector3d actual_pos_;
  Eigen::Vector3d actual_rpy_;
  // Store latest Jacobian for impedance computation
  Eigen::Matrix<double,6,7> J_eigen_;
  // ===================================================
};


int main(int argc,char**argv){
  rclcpp::init(argc,argv);
  auto node=std::make_shared<JointSpacePositionController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
