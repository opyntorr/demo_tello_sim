#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import tf_transformations
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

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
        
        # Suscriptor a los datos del OptiTrack
        self.opti_sub = self.create_subscription(
            PoseStamped,
            '/drone_pose',
            self.optitrack_callback,
            10
        )
        
        # Publicador de OptiTrack con covarianza
        self.opti_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/drone1/optitrack/pose_with_cov',
            10
        )
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

    def optitrack_callback(self, msg):
        # Crear el mensaje con covarianza
        pose_cov_msg = PoseWithCovarianceStamped()
        pose_cov_msg.header.stamp = self.get_clock().now().to_msg()
        pose_cov_msg.header.frame_id = "odom"
        
        # Intercambiar ejes Y y Z (Dado que OptiTrack usa Y hacia arriba)
        pose_cov_msg.pose.pose.position.x = msg.pose.position.x
        pose_cov_msg.pose.pose.position.y = msg.pose.position.z
        pose_cov_msg.pose.pose.position.z = msg.pose.position.y
        
        # Orientación (no se usa, establecer a identidad)
        pose_cov_msg.pose.pose.orientation.x = 0.0
        pose_cov_msg.pose.pose.orientation.y = 0.0
        pose_cov_msg.pose.pose.orientation.z = 0.0
        pose_cov_msg.pose.pose.orientation.w = 1.0
        
        # Covarianza (Matriz 6x6, 36 elementos)
        # [0]=X, [7]=Y, [14]=Z, [21]=Roll, [28]=Pitch, [35]=Yaw
        # Confianza muy alta en la posición (covarianza muy pequeña)
        pose_cov_msg.pose.covariance[0] = 0.0001
        pose_cov_msg.pose.covariance[7] = 0.0001
        pose_cov_msg.pose.covariance[14] = 0.0001
        
        # Confianza nula en la orientación (covarianza inmensa) porque no nos importa
        pose_cov_msg.pose.covariance[21] = 999.9
        pose_cov_msg.pose.covariance[28] = 999.9
        pose_cov_msg.pose.covariance[35] = 999.9
        
        self.opti_pub.publish(pose_cov_msg)

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
