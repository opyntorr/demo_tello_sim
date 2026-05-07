#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class ReferenceNode(Node):

    def __init__(self):
        super().__init__('tello_reference')

        self.declare_parameter('radius', 0.5)
        self.declare_parameter('frequency', 0.1)
        self.declare_parameter('height', 0.50)

        self.radius = self.get_parameter('radius').value
        self.frequency = self.get_parameter('frequency').value
        self.height = self.get_parameter('height').value

        self.omega = 2.0 * math.pi * self.frequency

        self.pub = self.create_publisher(Odometry, 'reference', 10)
        self.timer = self.create_timer(0.1, self.cb_timer)

        self.start_time = self.get_clock().now()
        self.get_logger().info(
            f'Reference node ready — radius={self.radius} m, '
            f'frequency={self.frequency} Hz, height={self.height} m'
        )

    def cb_timer(self):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = self.radius * math.cos(self.omega * t)
        msg.pose.pose.position.y = self.radius * math.sin(self.omega * t)
        msg.pose.pose.position.z = self.height
        msg.twist.twist.linear.x = -self.radius * self.omega * math.sin(self.omega * t)
        msg.twist.twist.linear.y =  self.radius * self.omega * math.cos(self.omega * t)
        msg.twist.twist.linear.z = 0.0
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
