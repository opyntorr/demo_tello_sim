#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class ControllerNode(Node):

    def __init__(self):
        super().__init__('tello_controller')

        self.declare_parameter('kp_x', 20.0)
        self.declare_parameter('kp_y', 20.0)
        self.declare_parameter('kp_z', 20.0)
        self.declare_parameter('max_speed', 100.0)

        self.kp_x = self.get_parameter('kp_x').value
        self.kp_y = self.get_parameter('kp_y').value
        self.kp_z = self.get_parameter('kp_z').value
        self.max_speed = self.get_parameter('max_speed').value

        self.pose = None
        self.reference = None
        self.yaw = 0.0

        self.sub_pose = self.create_subscription(Odometry, 'drone/pose', self.cb_pose, 10)
        self.sub_ref = self.create_subscription(Odometry, 'reference', self.cb_reference, 10)
        self.pub_control = self.create_publisher(Twist, 'cmd_vel', 10)

        self.timer = self.create_timer(0.05, self.cb_control)
        self.get_logger().info('Controller node ready')

    def cb_pose(self, msg):
        self.pose = msg
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def cb_reference(self, msg):
        self.reference = msg

    def clamp(self, value, limit):
        return max(-limit, min(limit, value))

    def cb_control(self):
        if self.pose is None or self.reference is None:
            return

        # Error in world frame
        ex = self.pose.pose.pose.position.x - self.reference.pose.pose.position.x
        ey = self.pose.pose.pose.position.y - self.reference.pose.pose.position.y
        ez = self.pose.pose.pose.position.z - self.reference.pose.pose.position.z

        # Control in world frame (P + feedforward)
        vx_w = -self.kp_x * ex + self.reference.twist.twist.linear.x
        vy_w = -self.kp_y * ey + self.reference.twist.twist.linear.y
        vz   = -self.kp_z * ez + self.reference.twist.twist.linear.z

        # Rotate from world frame to body frame
        vx = vx_w * math.cos(self.yaw) + vy_w * math.sin(self.yaw)
        vy = -vx_w * math.sin(self.yaw) + vy_w * math.cos(self.yaw)

        cmd = Twist()
        cmd.linear.x = float(self.clamp(vx, self.max_speed))
        cmd.linear.y = float(self.clamp(vy, self.max_speed))
        cmd.linear.z = float(self.clamp(vz, self.max_speed))
        cmd.angular.z = 0.0
        self.pub_control.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
