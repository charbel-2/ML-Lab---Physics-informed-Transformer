# my_launch_pkg/launch/my_nodes.launch.py
import os
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    cpp_node = Node(
            package= 'mujoco_joint_commander_cpp',
            executable= 'Cartesian_impedance_controller',
            name='impedance_node',
            output = 'screen',
        )
    
    python_node = Node(
        package= 'reference_generator',
        executable='reference_generator',
        name= 'reference_generator_node',
        output = 'screen'
    )

    delayed_python_node = TimerAction(
        period=0.5,  # seconds to wait
        actions=[python_node]
    )

    return LaunchDescription([
        cpp_node,
        delayed_python_node,
    ])