import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    directory = get_package_share_directory('mujoco_ros2')                                          # Gets relative path of mujoco_ros2 package    
    
    xmlScenePath =  '/home/polimi-panda/mujoco_models/franka_panda/panda_nohand_ros2.xml'
    
    if not os.path.exists(xmlScenePath):
        raise FileNotFoundError(f"Scene file does not exist: {xmlScenePath}.")


    mujoco = Node(
        package    = "mujoco_ros2",
        executable = "mujoco_node",
        output     = "screen",
        arguments  = [xmlScenePath],
        parameters = [   
    			{"joint_state_topic_name" : "joint_state"},
    			{"joint_command_topic_name" : "joint_commands"},
    			{"control_mode" : "TORQUE"},
    			{"simulation_frequency" : 1000},
    			{"visualisation_frequency" : 30},
    			{"camera_focal_point": [0.0, 0.0, 0.27]},
    			{"camera_distance": 2.7},
    			{"camera_azimuth": -135.0},
    			{"camera_elevation": -20.0},
    			{"camera_orthographic": False}
]

    )

    return LaunchDescription([mujoco])
    

