#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>    
#include <std_msgs/msg/float64_multi_array.hpp>
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

// MuJoCo 3.x
extern "C" {
  #include <mujoco.h>
}

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
      first_impedance_cycle_(true),     
      have_prev_desired_q_imp_(false),  
      have_prev_desired_pose_(false)  ,
      first_cmd_(true)  
  {
    // Parameters
    declare_parameter<std::string>("urdf_path",
      "/home/charbel/ros2_ws/src/robot_descriptions/urdf/panda_no_gripper.urdf");
    declare_parameter<std::string>("base_link", "fer_link0"); // if using panda.urdf: panda_link0 or panda_link8
    declare_parameter<std::string>("tip_link",  "fer_link8");

    declare_parameter<double>("ewma_alpha", 0.99);     // joint-space filter alpha
    declare_parameter<double>("damp_lambda", 0.05);    // damping for mapping back

    declare_parameter<std::string>("mujoco_xml_path",
      "/home/charbel/mujoco_models/franka_panda/panda_nohand_ros2.xml");

    declare_parameter<double>("cmd_ewma_alpha", 0.9);  // closer to 1 = smoother/slower response



    // Fill parameters in place
    get_parameter("urdf_path", urdf_path_);
    get_parameter("base_link", base_link_);
    get_parameter("tip_link",  tip_link_);
    get_parameter("ewma_alpha", ewma_alpha_);
    get_parameter("damp_lambda", damp_lambda_);
    get_parameter("mujoco_xml_path", mujoco_xml_path_);
    get_parameter("cmd_ewma_alpha", cmd_ewma_alpha_);


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
    prev_desired_q_imp_ = KDL::JntArray(nj);

    qddot_.resize(nj);
    jacobian_ = KDL::Jacobian(nj);
    q_kdl_.resize(nj);
    dyn_param_ = std::make_shared<KDL::ChainDynParam>(kdl_chain_, KDL::Vector(0,0,-9.81));
    tau_g_ = KDL::JntArray(nj);

    frozen_q_ = KDL::JntArray(nj);

    // Initialize filters
    // ewma_alpha_ = std::clamp(ewma_alpha_, 0.0, 1.0);
    damp_lambda_ = std::max(0.0, damp_lambda_);
    filtered_tau_ext_.setZero(nj);
    have_raw_wrench_ = false;
    first_tau_ext_ = true;

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
    Kp_joint_(0) = 500.;  // to double. 
    Kp_joint_(1) = 750.;  // 750.
    Kp_joint_(2) = 750.;  // 750.
    Kp_joint_(3) = 750.;  // 750.
    Kp_joint_(4) = 200.;
    Kp_joint_(5) = 100.;
    Kp_joint_(6) = 50.;
    //all values were half before
    Kd_joint_(0) = 64.;   // 32.
    Kd_joint_(1) = 80.;   // 40.
    Kd_joint_(2) = 60.;   // 30.
    Kd_joint_(3) = 90;  // 45.
    Kd_joint_(4) = 40.;   // 20.
    Kd_joint_(5) = 40.;   // 20.
    Kd_joint_(6) = 20.;    // 10. rimasto 10 nella prova

    Ki_joint_(0) = 45;  // 15.
    Ki_joint_(1) = 45.; // 15.
    Ki_joint_(2) = 60.;  // 20.
    Ki_joint_(3) = 120.;  // 40.
    Ki_joint_(4) = 120.;  // 40.
    Ki_joint_(5) = 90.;  // 30. rimasto 30 nella prova
    Ki_joint_(6) = 120.;  // 40. rimasto 40 nelòla prova

    

    // Init previous dq/time
    prev_time_ = now();
    for (size_t i = 0; i < nj; ++i) {
      prev_dq_(i) = 0.0;
      qddot_(i)   = 0.0;
    }
    last_dt_ = 0.001;

    // Initialize desired-pose velocity tracking
    desired_pos_.setZero();
    desired_rpy_.setZero();
    prev_desired_pos_.setZero();  
    prev_desired_rpy_.setZero();  

    // Subscribers / Publishers

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_state", 10,
      std::bind(&JointSpacePositionController::jointStateCallback, this, std::placeholders::_1)
    );
    desired_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/optitrack_pose", 10,
      std::bind(&JointSpacePositionController::desiredPoseCallback, this, std::placeholders::_1)
    );
    ee_wrench_sub_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/ee_wrench_sim", 10,
      std::bind(&JointSpacePositionController::eeWrenchCallback, this, std::placeholders::_1)
    );
    
    // /ee_contact_wrench includes the cleaned "interaction" external forces at ee
    // /ee_contact_wrench_filtered is /ee_contact_wrench with a filter ---> this is the one to use for controllers
    // For higher level controllers please subscribe to topics from this code:
    // actual position ---> actual_ee_pose (PoseStamped)
    // desired pisittion ---> desired_ee_pose (PoseStamped)
    // interaction force --> ee_contact_wrench_filtered (WrenchStamped)
    // impedance_reference_pose --> new ee pose from impedance (PoseStamped)

    joint_command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("joint_commands", 1);

    // ee_contact_wrench_pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
    //   "/ee_contact_wrench", 10);
    
    ee_contact_wrench_filtered_pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
      "/ee_contact_wrench_filtered", 10);

    desired_ee_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "desired_ee_pose", 10);
    actual_ee_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "actual_ee_pose", 10);

    actual_ee_velocity_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
      "actual_ee_velocity", 10);

    impedance_ref_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "impedance_reference_pose", 10);
    // ==========================================================

    // virtual impedance parameters:
    // Use values from georges's code to make the robot jump or oscillate like a spring, maybe to test the controller
    mass_imp_ = 10.0;               // 5
    inertia_imp_ = 10.0;  // 5
    stiffness_trans_ = 200.0; // 100
    stiffness_rot_ = 200.0; // 100
    damping_factor_trans_ = 10; // 1
    damping_factor_rot_ = 10; // 1
    
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

  // Update acceleration by finite difference
  void updateAcceleration(rclcpp::Time now) {
    // <<< MODIFIED: use real dt instead of fixed 1/1000
    double dt = (now - prev_time_).seconds();
    if (dt <= 0) {
      // Keep last_dt_ unchanged if time did not advance
      return;
    }
    for (int i = 0; i < (int)qddot_.rows(); ++i) {
      qddot_(i) = (current_dq_(i) - prev_dq_(i)) / dt;
      prev_dq_(i) = current_dq_(i);
    }
    prev_time_ = now;

    last_dt_ = dt;
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
        RCLCPP_WARN(get_logger(), "IK solver failed in desiredPoseCallback");
      }
    }
    // store desired EE pose for impedance 
    // Also track previous desired pose to compute desired velocity
    if (have_prev_desired_pose_) {
      // prev_desired_pos_ and prev_desired_rpy_ already set
    } else {
      // first time: set prev = current desired
      prev_desired_pos_.x() = x;
      prev_desired_pos_.y() = y;
      prev_desired_pos_.z() = z;
      prev_desired_rpy_.x() = roll;
      prev_desired_rpy_.y() = pitch;
      prev_desired_rpy_.z() = yaw;
      have_prev_desired_pose_ = true;
    }
    desired_pos_.x() = x;
    desired_pos_.y() = y;
    desired_pos_.z() = z;
    desired_rpy_.x() = roll;
    desired_rpy_.y() = pitch;
    desired_rpy_.z() = yaw;

    // publish desired EE pose 
    {
      geometry_msgs::msg::PoseStamped desired_msg;
      // copy header from incoming msg, so frame_id and stamp are consistent
      desired_msg.header = msg->header;
      desired_msg.pose = msg->pose;
      desired_ee_pose_pub_->publish(desired_msg);
    }

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

            // Publish as TwistStamped
            geometry_msgs::msg::TwistStamped vel_msg;
            vel_msg.header = msg->header;
            // linear
            vel_msg.twist.linear.x  = twist(0);
            vel_msg.twist.linear.y  = twist(1);
            vel_msg.twist.linear.z  = twist(2);
            // angular
            vel_msg.twist.angular.x = twist(3);
            vel_msg.twist.angular.y = twist(4);
            vel_msg.twist.angular.z = twist(5);
            actual_ee_velocity_pub_->publish(vel_msg);
          } else {
            RCLCPP_WARN(get_logger(), "KDL Jacobian computation failed for velocity");
          }
        }
      } else {
        RCLCPP_WARN(get_logger(), "FK solver failed to compute actual EE pose");
      }
    }
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


    }

    //  Impedance offset dynamics (anchored around moving desired) and compute impedance reference pose and velocity

    // Use KDL::JntArray for desired_q_imp:
    KDL::JntArray desired_q_imp_(nj);
    KDL::JntArray desired_dq_imp_(nj);

    // Default fallback: desired_q_imp_ = desired_q_, desired_dq_imp_ = zero
    // for (size_t i = 0; i < nj; ++i) {
    //   desired_q_imp_(i) = desired_q_(i);
    //   desired_dq_imp_(i) = 0.0;
    // }

    if (have_impedance) {
      // 1) Update offset dynamics: M * ddot_offset + D * dot_offset + K * offset = F_filt
      // offset = imp_pos - desired_pos
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
      
      // --- POSITION IK only for initialization (first cycle) ---
      if (first_impedance_cycle_) {
        // On the very first impedance cycle, compute initial desired_q_imp_ from Cartesian impedance pose
        if (ik_solver_) {
          KDL::Frame target_imp(
            KDL::Rotation::RPY(imp_rpy.x(), imp_rpy.y(), imp_rpy.z()),
            KDL::Vector(imp_pos.x(), imp_pos.y(), imp_pos.z())
          );
          KDL::JntArray q0(kdl_chain_.getNrOfJoints());
          for (size_t i = 0; i < q0.rows(); ++i) {
            q0(i) = current_q_(i);
          }
          // prev_desired_q_imp_ = KDL::JntArray(nj); // ensure correct size

          int status = ik_solver_->CartToJnt(q0, target_imp, prev_desired_q_imp_);
          if (status < 0) {
            RCLCPP_WARN(get_logger(), "Impedance IK solver failed; using previous desired_q_imp_");
            // desired_q_imp_ remains as previous desired_q_imp_
            for (size_t i = 0; i < nj; ++i) {
              prev_desired_q_imp_(i) = current_q_(i);
            }
          }
        }
        else {
          // No IK solver? fallback
          // prev_desired_q_imp_ = KDL::JntArray(nj);
          for (size_t i = 0; i < nj; ++i) {
            prev_desired_q_imp_(i) = current_q_(i);
          }
        }
        have_prev_desired_q_imp_ = true;
        first_impedance_cycle_ = false;
      }

      // Compute desired EE velocity from desired_pos_ and prev_desired_pos_
      Eigen::Vector3d des_vel_t = Eigen::Vector3d::Zero();
      Eigen::Vector3d des_vel_r = Eigen::Vector3d::Zero();
      if (have_prev_desired_pose_) {
        double dt_pose = last_dt_;
        if (dt_pose > 1e-6) {
          des_vel_t = (desired_pos_ - prev_desired_pos_) / dt_pose;
          des_vel_r = (desired_rpy_ - prev_desired_rpy_) / dt_pose;
        }
      }

      // Update prev_desired for next cycle
      prev_desired_pos_ = desired_pos_;
      prev_desired_rpy_ = desired_rpy_;
      have_prev_desired_pose_ = true;

      Eigen::Vector3d imp_vel_t = des_vel_t + offset_vel_.head<3>();
      Eigen::Vector3d imp_vel_r = des_vel_r  + offset_vel_.tail<3>();

      // 4) Compute velocity IK for impedance reference twist to get desired_dq_imp_
      bool did_velocity_ik = false;

      if (vel_ik_solver_) {
        // Build desired twist: linear + angular velocities.
        // Note: imp_vel_t is linear velocity (m/s), imp_vel_r is angular velocity (rad/s).
        KDL::Twist desired_twist(
          KDL::Vector(imp_vel_t.x(), imp_vel_t.y(), imp_vel_t.z()),
          KDL::Vector(imp_vel_r.x(), imp_vel_r.y(), imp_vel_r.z())
        );
        KDL::JntArray q0_vel(nj);
        if (have_prev_desired_q_imp_) {
          for (size_t i = 0; i < nj; ++i) {
            q0_vel(i) = prev_desired_q_imp_(i);
          }
        } else {
          for (size_t i = 0; i < nj; ++i) {
            q0_vel(i) = current_q_(i);
          }
        }
        int vstatus = vel_ik_solver_->CartToJnt(q0_vel, desired_twist, desired_dq_imp_);
        if (vstatus < 0) {
          RCLCPP_WARN(get_logger(), "Velocity IK solver failed; setting desired_dq_imp_ to zero");
          for (size_t i = 0; i < nj; ++i) {
            desired_dq_imp_(i) = 0.0;
          }
        } else {
          did_velocity_ik = true;
        }
      } else {
        // No velocity IK solver? set zero
        for (size_t i = 0; i < nj; ++i) {
          desired_dq_imp_(i) = 0.0;
        }
      }


      // --- INTEGRATE desired_q_imp_ from desired_dq_imp_ ---
      if (did_velocity_ik && have_prev_desired_q_imp_) {
        double dt_int = last_dt_;
        for (size_t i = 0; i < nj; ++i) {
          desired_q_imp_(i) = prev_desired_q_imp_(i) + desired_dq_imp_(i) * dt_int;
        }
        // Optionally: clamp desired_q_imp_ within joint limits here
      } else {
        // If velocity IK failed or no prev: fallback to previous desired_q_imp_ or current_q_
        if (have_prev_desired_q_imp_) {
          for (size_t i = 0; i < nj; ++i) {
            desired_q_imp_(i) = prev_desired_q_imp_(i);
          }
        } else {
          for (size_t i = 0; i < nj; ++i) {
            desired_q_imp_(i) = current_q_(i);
          }
        }
      }

      //  save for next cycle
      prev_desired_q_imp_ = desired_q_imp_;
      have_prev_desired_q_imp_ = true;
      

    }

    else {
      // // No impedance: fallback desired to nominal desired_q_ and zero vel
      // for (size_t i = 0; i < nj; ++i) {
      //   desired_q_imp_(i) = desired_q_(i);
      //   desired_dq_imp_(i) = 0.0;
      // }
      // Reset impedance state so that next impedance start re-initializes
      first_impedance_cycle_ = true;
      have_prev_desired_q_imp_ = false;
      // Also reset desired-pose velocity tracking so next trajectory starts fresh
      have_prev_desired_pose_ = false;
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
      Eigen::Matrix<double,7,1> tau_pid = Kp_joint_.cwiseProduct(e_q)
                                       + Kd_joint_.cwiseProduct(e_dq)
                                       + Ki_joint_.cwiseProduct(integral_joint_error_);
      Eigen::Matrix<double,7,1> tau_cmd = saturateTorqueRate(tau_pid);


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

  // === ADDITION: publishers for EE poses and impedance ref ===
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr desired_ee_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr actual_ee_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr actual_ee_velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr impedance_ref_pose_pub_;
  // ==================================================

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

  Eigen::Vector3d prev_desired_pos_;  
  Eigen::Vector3d prev_desired_rpy_;  
  // Store latest Jacobian for impedance computation
  Eigen::Matrix<double,6,7> J_eigen_;


  bool first_impedance_cycle_;         // to initialize desired_q_imp_ once
  bool have_prev_desired_q_imp_;       // whether prev_desired_q_imp_ is valid
  bool have_prev_desired_pose_;       
  KDL::JntArray prev_desired_q_imp_;   // store previous-cycle joint impedance target
  // ===================================================
};

int main(int argc,char**argv){
  rclcpp::init(argc,argv);
  auto node=std::make_shared<JointSpacePositionController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

