import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import math
from rclpy.signals import SignalHandlerOptions

class TelloPositionController(Node):
    def __init__(self):
        super().__init__('tello_position_controller')
        
        # Publicadores y Suscriptores
        self.cmd_vel_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/drone1/integrated_odom', self.odom_callback, 10)
        self.target_sub = self.create_subscription(Point, '/drone1/target_position', self.target_callback, 10)
        
        # Posición objetivo inicializada en None (esperando comando)
        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.target_received = False
        
        # Ganancias del controlador PID (Perfil muy lento)
        self.kp = 0.15  # Empuje inicial muy suave
        self.ki = 0.01  # Corrección a largo plazo mínima
        self.kd = 0.35  # Freno aerodinámico fuerte para evitar oscilaciones
        
        # Límites de saturación
        self.max_vel = 0.15          # Velocidad máxima estricta (15 cm/s)
        self.max_integral = 0.5      # Límite bajo de acumulación
        
        # Memoria de estado para derivadas e integrales
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_z = 0.0
        
        self.current_pose = None
        self.last_time = None
        self.start_time = None
        
        # Bucle de control a 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)
        
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position
        
        
    def target_callback(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y
        self.target_z = msg.z
        self.target_received = True
        self.get_logger().info(f"🎯 Nuevo objetivo recibido: X={self.target_x}, Y={self.target_y}, Z={self.target_z}")
        
    def control_loop(self):
        if self.current_pose is None or not self.target_received:
            return
            
        current_time = self.get_clock().now()
        
        # Inicializar el tiempo en la primera ejecución
        if self.last_time is None:
            self.last_time = current_time
            self.start_time = current_time
            return
            
        # Calcular delta de tiempo real en segundos
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
            
        # 1. Calcular Errores actuales (P)
        error_x = self.target_x - self.current_pose.x
        error_y = self.target_y - self.current_pose.y
        error_z = self.target_z - self.current_pose.z
        
        # Distancia euclidiana al objetivo
        distance = math.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        twist = Twist()
        
        # Tolerancia para detenerse (10 cm)
        if distance > 0.10: 
            
            # 2. Calcular Integrales (I) y aplicar Anti-windup
            self.integral_x += error_x * dt
            self.integral_y += error_y * dt
            self.integral_z += error_z * dt
            
            self.integral_x = max(min(self.integral_x, self.max_integral), -self.max_integral)
            self.integral_y = max(min(self.integral_y, self.max_integral), -self.max_integral)
            self.integral_z = max(min(self.integral_z, self.max_integral), -self.max_integral)
            
            # 3. Calcular Derivadas (D)
            derivative_x = (error_x - self.prev_error_x) / dt
            derivative_y = (error_y - self.prev_error_y) / dt
            derivative_z = (error_z - self.prev_error_z) / dt
            
            # 4. Ecuación PID
            vel_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * derivative_x)
            vel_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * derivative_y)
            vel_z = (self.kp * error_z) + (self.ki * self.integral_z) + (self.kd * derivative_z)
            
            # 5. Aplicar saturación (Clamp de velocidad)
            twist.linear.x = max(min(vel_x, self.max_vel), -self.max_vel)
            twist.linear.y = max(min(vel_y, self.max_vel), -self.max_vel)
            twist.linear.z = max(min(vel_z, self.max_vel), -self.max_vel)
            
        else:
            # Detener el dron al alcanzar el objetivo y resetear integrales
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            self.integral_x = 0.0
            self.integral_y = 0.0
            self.integral_z = 0.0
            self.get_logger().info("¡Posición objetivo alcanzada de forma estable!", throttle_duration_sec=2.0)
            
        # Guardar estado para la siguiente iteración
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_z = error_z
        self.last_time = current_time
            
        # Publicar el comando de velocidad
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TelloPositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Deteniendo controlador...")
    finally:
        stop_msg = Twist()
        node.cmd_vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()