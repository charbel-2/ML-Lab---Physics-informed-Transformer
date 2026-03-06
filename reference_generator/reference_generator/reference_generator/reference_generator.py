#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from numpy import random
import numpy as np
import math

class ReferenceGenerator(Node):
    def __init__(self):
        super().__init__('reference_generator')

        # Publisher for desired end-effector pose as PoseStamped
        self.publisher_ = self.create_publisher(
            PoseStamped,
            '/optitrack_pose',
            10  # QoS history depth
        )

        # Publisher for desired end-effector pose as PoseStamped
        self.publisher_future = self.create_publisher(
            PoseStamped,
            '/optitrack_pose_future',
            10  # QoS history depth
        )

        # Publisher for desired end-effector pose as PoseStamped
        self.force_publisher = self.create_publisher(
            WrenchStamped,
            '/desired_force',
            10  # QoS history depth
        )

        # Publish at 1000 Hz
        self.timer_period = 0.001  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.start_time = self.get_clock().now()

        self.get_logger().info('Reference generator node has been started.')

    def timer_callback(self):
        now = self.get_clock().now()
        elapsed_time = (now - self.start_time).nanoseconds / 1e9  # seconds
        future_elapsed_time = elapsed_time + (240*self.timer_period)

        # 1) Base desired position (x, y, z)
        desired_position = np.array([0.6, 0.0, 0.61])
        desired_position_future = np.array([0.6, 0.0, 0.61])

        desired_force = np.array([0, 0, 8.0])


        # 2) Desired orientation as Euler angles (roll, pitch, yaw)
        #    We want the EE “perpendicular to ground,” so pitch = π
        roll  = 0.0
        pitch = math.pi
        yaw   = 0.0

        # 3) Convert Euler → quaternion (ZYX convention)
        cy = math.cos(yaw   * 0.5)
        sy = math.sin(yaw   * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll  * 0.5)
        sr = math.sin(roll  * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy







        # 4) Build and publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now.to_msg()
        pose_msg.header.frame_id = 'panda_link0'  # must match your KDL base frame

        pose_msg.pose.position.x = float(desired_position[0])
        pose_msg.pose.position.y = float(desired_position[1])
        pose_msg.pose.position.z = float(desired_position[2])

        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.publisher_.publish(pose_msg)


        future_pose_msg =PoseStamped()
        future_pose_msg.header.stamp =now.to_msg()
        future_pose_msg.header.frame_id = 'panda_link0'

        future_pose_msg.pose.position.x = float(desired_position_future[0])
        future_pose_msg.pose.position.y = float(desired_position_future[1])
        future_pose_msg.pose.position.z = float(desired_position_future[2])

        future_pose_msg.pose.orientation.x = qx
        future_pose_msg.pose.orientation.y = qy
        future_pose_msg.pose.orientation.z = qz
        future_pose_msg.pose.orientation.w = qw

        self.publisher_future.publish(future_pose_msg)

        desired_force_msg = WrenchStamped()
        desired_force_msg.header.stamp = now.to_msg()
        desired_force_msg.header.frame_id = 'end_effector'

        desired_force_msg.wrench.force.x = float(desired_force[0])
        desired_force_msg.wrench.force.y = float(desired_force[1])
        desired_force_msg.wrench.force.z = float(desired_force[2])
        desired_force_msg.wrench.torque.x = float(np.random.uniform(-0.25, 0.25))
        desired_force_msg.wrench.torque.y = float(np.random.uniform(-0.25, 0.25))
        desired_force_msg.wrench.torque.z = float(np.random.uniform(-0.25, 0.25))

        self.force_publisher.publish(desired_force_msg)


        self.get_logger().info(
            f'Published desired EE pose and future desired pose: '
            f'x={pose_msg.pose.position.x:.3f}, y={pose_msg.pose.position.y:.3f}, '
            f'z={pose_msg.pose.position.z:.3f}, '
            f'roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}, '
            f'desired force = {desired_force[2]}'
        )





def main(args=None):
    rclpy.init(args=args)
    node = ReferenceGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
