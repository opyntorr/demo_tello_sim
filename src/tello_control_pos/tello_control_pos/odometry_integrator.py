#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import tf_transformations

class SimpleIntegratorOdom(Node):
    def __init__(self):
        super().__init__('simple_integrator_odom')
        
        # Suscriptor al SENSOR de velocidad del dron (En Gazebo viene dentro de Odom, 
        # en el dron real vendría de /flight_data)
        self.vel_sub = self.create_subscription(
            Odometry, 
            '/drone1/odom', 
            self.velocity_sensor_callback, 
            10
        )
        
        # Publicador de la nueva odometría integrada
        self.odom_pub = self.create_publisher(
            Odometry, 
            '/drone1/integrated_odom', 
            10
        )
        
        # Variables de estado (Posición actual)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.theta = 0.0
        
        # Velocidades sensadas actuales (en m/s reales)
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.w_z = 0.0
        
        self.last_time = self.get_clock().now()
        
        # Bucle de integración a alta frecuencia (50 Hz)
        self.timer = self.create_timer(0.02, self.integration_loop)
        
    def velocity_sensor_callback(self, msg):
        # Leemos el SENSOR de velocidad. (Twist viene en el Body Frame)
        # NOTA: El driver del dron real divide por 100.0 (en vez de 10.0) 
        # las velocidades en dm/s, haciéndolas 10 veces más pequeñas. 
        # Multiplicamos por 10.0 aquí para corregirlo sin tocar el driver.
        self.v_x = msg.twist.twist.linear.x * 10.0
        self.v_y = msg.twist.twist.linear.y * 10.0
        self.v_z = msg.twist.twist.linear.z * 10.0
        self.w_z = msg.twist.twist.angular.z
        
        # Leemos la altura absoluta (TOF) si está disponible
        incoming_z = msg.pose.pose.position.z
        # Filtrar picos de error del sensor (ej. 65 metros cuando pierde señal)
        if 0.01 < incoming_z < 5.0:
            self.z = incoming_z

    def integration_loop(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        if dt <= 0.0:
            return
            
        # 1. Modelo Cinemático del Dron (Integración Euler Simple)
        # Aquí ya NO necesitamos multiplicadores, porque estamos integrando la 
        # lectura directa del SENSOR de velocidad (que ya viene en m/s puros)
        
        self.theta += self.w_z * dt
        
        # Rotación 2D para pasar de Body Frame a World Frame
        world_v_x = self.v_x * math.cos(self.theta) - self.v_y * math.sin(self.theta)
        world_v_y = self.v_x * math.sin(self.theta) + self.v_y * math.cos(self.theta)
        
        # Integración Numérica (x = x + v * dt)
        self.x += world_v_x * dt
        self.y += world_v_y * dt
        
        # NOTA: La Z ya no la integramos porque ahora usamos el sensor TOF absoluto 
        # que seteamos en el callback. (Si z = 0, se queda en 0 hasta que suba).
        
        # 2. Publicar la Odometría
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        
        # Posición
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = self.z
        
        # Orientación (Convertir theta a cuaternión)
        q = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        
        # Velocidades (Guardamos las globales por completitud)
        odom_msg.twist.twist.linear.x = world_v_x
        odom_msg.twist.twist.linear.y = world_v_y
        odom_msg.twist.twist.linear.z = self.v_z
        odom_msg.twist.twist.angular.z = self.w_z
        
        self.odom_pub.publish(odom_msg)
        self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = SimpleIntegratorOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
