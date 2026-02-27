#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/frames.hpp>

#include <urdf/model.h>
#include <fstream>
#include <string>

#include <Eigen/Dense>

class JointSpacePositionController : public rclcpp::Node {
public:
  JointSpacePositionController()
    : Node("joint_space_position_controller"),
      desired_q_(7), prev_dq_(7), qddot_(7),
      integral_joint_error_(7), last_torque_(7), filtered_wrench_(6)
  {
    // Parameters
    declare_parameter<std::string>("urdf_path",
      "/home/charbel/ros2_ws/src/robot_descriptions/urdf/fr3.urdf");
    declare_parameter<std::string>("base_link", "base");
    declare_parameter<std::string>("tip_link",  "fr3_link7");
    get_parameter("urdf_path", urdf_path_);
    get_parameter("base_link", base_link_);
    get_parameter("tip_link",  tip_link_);

    if (!loadURDFFile(urdf_path_, urdf_xml_)) {
      RCLCPP_ERROR(get_logger(), "Failed to load URDF from '%s'", urdf_path_.c_str());
      rclcpp::shutdown(); return;
    }
    if (!initializeKDL(urdf_xml_)) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize KDL from URDF");
      rclcpp::shutdown(); return;
    }

    // Dynamics helper
    dyn_param_ = std::make_shared<KDL::ChainDynParam>(kdl_chain_, KDL::Vector(0,0,-9.81));
    size_t nj = kdl_chain_.getNrOfJoints();
    tau_c_ = KDL::JntArray(nj);
    tau_g_ = KDL::JntArray(nj);
    H_     = KDL::JntSpaceInertiaMatrix(nj);

    // Finite diff buffer
    prev_time_ = now();

    // Weights for damped least-squares
    damp_lambda_ = 0.05;

    // Low-pass EWMA filter alpha
    ewma_alpha_ = 0.00;

    // PID gains
    Kp_joint_.setConstant(500.0);
    Kd_joint_.setConstant(20.0);
    Ki_joint_.setConstant(40.0);
    delta_tau_max_ = 20.0;

        // Initialize joint arrays and jacobian
    current_q_ = KDL::JntArray(nj);
    current_dq_ = KDL::JntArray(nj);
    desired_q_ = KDL::JntArray(nj);
    jacobian_   = KDL::Jacobian(nj);
    integral_joint_error_.setZero();
    last_torque_.setZero();
    // zero KDL-based velocity & acceleration buffers
    for (size_t i = 0; i < nj; ++i) {
      prev_dq_(i) = 0.0;
      qddot_(i)   = 0.0;
    }
    // zero Eigen wrench buffer
    filtered_wrench_.setZero();

    // Subs & pubs
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_state", 10,
      std::bind(&JointSpacePositionController::jointStateCallback, this, std::placeholders::_1)
    );
    desired_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/optitrack_pose", 10,
      std::bind(&JointSpacePositionController::desiredPoseCallback, this, std::placeholders::_1)
    );
    joint_command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("joint_commands", 1);
    end_effector_pose_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/end_effector_pose", 1);
    ee_contact_wrench_pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
      "/ee_contact_wrench", 10);
    ee_contact_wrench_filtered_pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
      "/ee_contact_wrench_filtered", 10);

    RCLCPP_INFO(get_logger(), "Controller with improved force estimation started.");
  }

private:
  bool loadURDFFile(const std::string &path, std::string &urdf_xml) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    urdf_xml.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return true;
  }

  bool initializeKDL(const std::string &urdf_xml) {
    urdf::Model model;
    if (!model.initString(urdf_xml)) return false;
    KDL::Tree tree;
    if (!kdl_parser::treeFromUrdfModel(model, tree)) return false;
    if (!tree.getChain(base_link_, tip_link_, kdl_chain_)) return false;
    fk_solver_  = std::make_shared<KDL::ChainFkSolverPos_recursive>(kdl_chain_);
    jac_solver_ = std::make_shared<KDL::ChainJntToJacSolver>(kdl_chain_);
    ik_solver_  = std::make_shared<KDL::ChainIkSolverPos_LMA>(kdl_chain_);
    return true;
  }

  void updateAcceleration(rclcpp::Time now) {
    double dt = (now - prev_time_).seconds();
    if (dt <= 0) return;
    for (int i=0; i<qddot_.rows(); ++i) {
      qddot_(i) = (current_dq_(i) - prev_dq_(i)) / dt;
      prev_dq_(i) = current_dq_(i);
    }
    prev_time_ = now;
  }

  Eigen::Matrix<double,7,1> saturateTorqueRate(const Eigen::Matrix<double,7,1> &tau_cmd) {
    Eigen::Matrix<double,7,1> tau_out;
    for (size_t i=0; i<7; ++i) {
      double d = tau_cmd(i) - last_torque_(i);
      d = std::clamp(d, -delta_tau_max_, delta_tau_max_);
      tau_out(i) = last_torque_(i) + d;
    }
    return tau_out;
  }

  void desiredPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    double x=msg->pose.position.x, y=msg->pose.position.y, z=msg->pose.position.z;
    double roll,pitch,yaw;
    quaternionToRPY(msg->pose.orientation.x,msg->pose.orientation.y,
                    msg->pose.orientation.z,msg->pose.orientation.w,
                    roll,pitch,yaw);
    KDL::Frame target(KDL::Rotation::RPY(roll,pitch,yaw), KDL::Vector(x,y,z));
    KDL::JntArray q0(kdl_chain_.getNrOfJoints());
    for (int i=0;i<q0.rows();++i) q0(i)=current_q_(i);
    ik_solver_->CartToJnt(q0, target, desired_q_);
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    auto now_time = now();
    if (msg->position.size()!=current_q_.rows()) return;
    for (int i=0; i<current_q_.rows(); ++i) {
      current_q_(i)  = msg->position[i];
      current_dq_(i) = msg->velocity[i];
    }
    updateAcceleration(now_time);

    //---- PID
    Eigen::Matrix<double,7,1> e_q, e_dq;
    for (int i=0; i<7; ++i) {
      e_q(i)   = desired_q_(i) - current_q_(i);
      e_dq(i)  = -current_dq_(i);
      integral_joint_error_(i) += e_q(i) * 0.001;
    }
    Eigen::Matrix<double,7,1> tau_pid = Kp_joint_.cwiseProduct(e_q)
                                     + Kd_joint_.cwiseProduct(e_dq)
                                     + Ki_joint_.cwiseProduct(integral_joint_error_);
    Eigen::Matrix<double,7,1> tau_cmd = saturateTorqueRate(tau_pid);
    std_msgs::msg::Float64MultiArray cmd; cmd.data.resize(7);
    for (int i=0;i<7;++i) { cmd.data[i]=tau_cmd(i); last_torque_(i)=tau_cmd(i); }
    joint_command_pub_->publish(cmd);

    //---- Force estimation
    Eigen::Matrix<double,7,1> tau_meas;
    for (int i=0; i<7; ++i) tau_meas(i)=msg->effort[i];

    dyn_param_->JntToMass(current_q_, H_);
    dyn_param_->JntToCoriolis(current_q_, current_dq_, tau_c_);
    dyn_param_->JntToGravity(current_q_, tau_g_);

    // tau_model = H*qddot + C + G
    Eigen::Map<Eigen::Matrix<double,7,1>> Hqdd(qddot_.data.data());
    Eigen::Matrix<double,7,1> tau_model = Hqdd+
      Eigen::Map<Eigen::Matrix<double,7,1>>(tau_c_.data.data())+
      Eigen::Map<Eigen::Matrix<double,7,1>>(tau_g_.data.data());

    Eigen::Matrix<double,7,1> tau_ext = tau_meas - tau_model;//tau_meas + tau_cmd + tau_model;//- tau_model;//tau_meas - 

    jac_solver_->JntToJac(current_q_, jacobian_);
    Eigen::Matrix<double,6,7> J; for(int r=0;r<6;++r) for(int c=0;c<7;++c) J(r,c)=jacobian_(r,c);

    // Damped least-squares: F_ext = J^T * (J*J^T + λ²I)^{-1} * tau_ext
    Eigen::Matrix<double,6,6> lambdaI = damp_lambda_*damp_lambda_ * Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,6> JJt = J * J.transpose() + lambdaI;
    Eigen::Matrix<double,6,1> tmp = -(J * tau_ext);
    Eigen::Matrix<double,6,1> F_ext = JJt.ldlt().solve(tmp);

    // Publish
    geometry_msgs::msg::WrenchStamped raw; raw.header=msg->header;
    raw.wrench.force.x = F_ext(0);
    raw.wrench.force.y = F_ext(1);
    raw.wrench.force.z = F_ext(2);
    raw.wrench.torque.x = F_ext(3);
    raw.wrench.torque.y = F_ext(4);
    raw.wrench.torque.z = F_ext(5);
    ee_contact_wrench_pub_->publish(raw);

    // EWMA filter
    filtered_wrench_ = ewma_alpha_ * F_ext + (1-ewma_alpha_)*filtered_wrench_;
    geometry_msgs::msg::WrenchStamped filt; filt.header = raw.header;
    filt.wrench.force.x = filtered_wrench_(0);
    filt.wrench.force.y = filtered_wrench_(1);
    filt.wrench.force.z = filtered_wrench_(2);
    filt.wrench.torque.x = filtered_wrench_(3);
    filt.wrench.torque.y = filtered_wrench_(4);
    filt.wrench.torque.z = filtered_wrench_(5);
    ee_contact_wrench_filtered_pub_->publish(filt);
  }

  void quaternionToRPY(double qx,double qy,double qz,double qw,double &roll,double &pitch,double &yaw) {
    Eigen::Quaterniond q(qw,qx,qy,qz); auto R=q.toRotationMatrix();
    roll=std::atan2(R(2,1),R(2,2));
    pitch=std::atan2(-R(2,0),std::sqrt(R(2,1)*R(2,1)+R(2,2)*R(2,2)));
    yaw=std::atan2(R(1,0),R(0,0));
  }

  // Members
  std::string urdf_path_, urdf_xml_, base_link_, tip_link_;
  KDL::Chain kdl_chain_;
  std::shared_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_;
  std::shared_ptr<KDL::ChainIkSolverPos_LMA> ik_solver_;
  std::shared_ptr<KDL::ChainDynParam> dyn_param_;

  KDL::JntArray current_q_, current_dq_, desired_q_, tau_c_, tau_g_;
  KDL::JntSpaceInertiaMatrix H_;
  KDL::Jacobian jacobian_;

  KDL::JntArray prev_dq_, qddot_;
  rclcpp::Time prev_time_;

  Eigen::Matrix<double,7,1> Kp_joint_, Kd_joint_, Ki_joint_, integral_joint_error_, last_torque_;
  double delta_tau_max_;

  double damp_lambda_, ewma_alpha_;
  Eigen::Matrix<double,6,1> filtered_wrench_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr desired_pose_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_pub_, end_effector_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ee_contact_wrench_pub_, ee_contact_wrench_filtered_pub_;
};

int main(int argc,char**argv){rclcpp::init(argc,argv);
  auto node=std::make_shared<JointSpacePositionController>();
  rclcpp::spin(node);
  rclcpp::shutdown();return 0;
}
