#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import tf_transformations

class CovarianceInjector(Node):
    def __init__(self):
        super().__init__('covariance_injector')
        
        # Suscriptor al SENSOR de velocidad del dron (En Gazebo viene dentro de Odom, 
        # en el dron real vendría de /flight_data)
        self.vel_sub = self.create_subscription(
            Odometry, 
            '/drone1/odom', 
            self.velocity_sensor_callback, 
            10
        )
        
        # Publicador de odometría con covarianzas para el EKF
        self.odom_pub = self.create_publisher(
            Odometry, 
            '/drone1/odom_with_cov', 
            10
        )

        
        # Multiplicador para corregir la escala de velocidad del sensor
        self.declare_parameter('vel_multiplier', 10.0)
        self.multiplier = self.get_parameter('vel_multiplier').get_parameter_value().double_value
        

        
    def velocity_sensor_callback(self, msg):
        # Usamos el multiplicador configurado (10.0 para real, 1.0 para simulación)
        raw_v_x = msg.twist.twist.linear.x * self.multiplier
        raw_v_y = msg.twist.twist.linear.y * self.multiplier
        raw_v_z = msg.twist.twist.linear.z * self.multiplier
        raw_w_z = msg.twist.twist.angular.z
        
        # Preparamos el mensaje Odometry para el EKF
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        
        # 1. Velocidades (Twist)
        odom_msg.twist.twist.linear.x = raw_v_x
        odom_msg.twist.twist.linear.y = raw_v_y
        odom_msg.twist.twist.linear.z = raw_v_z
        odom_msg.twist.twist.angular.z = raw_w_z
        
        # Covarianza de Velocidad (Basada en ruido_odometria.csv)
        # Matriz 6x6 (36 elementos)
        # [0]=X, [7]=Y, [14]=Z, [21]=Roll, [28]=Pitch, [35]=Yaw
        odom_msg.twist.covariance[0] = 0.005   # Varianza Vx
        odom_msg.twist.covariance[7] = 0.005   # Varianza Vy
        odom_msg.twist.covariance[14] = 0.001  # Varianza Vz
        odom_msg.twist.covariance[35] = 0.01   # Varianza Yaw
        
        # 2. Posición Z absoluta (Pose)
        incoming_z = msg.pose.pose.position.z
        if 0.01 < incoming_z < 5.0:
            odom_msg.pose.pose.position.z = incoming_z
            odom_msg.pose.covariance[14] = 0.001 # Varianza Z
        else:
            odom_msg.pose.pose.position.z = 0.0
            odom_msg.pose.covariance[14] = 999.9 # Ignorar Z si es inválida
            
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CovarianceInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
