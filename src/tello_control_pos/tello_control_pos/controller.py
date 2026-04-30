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
        
        # Ganancias del controlador PID
        self.kp = 0.8   # Aumentado para superar la inercia del dron real
        self.ki = 0.02  # Corrección a largo plazo
        self.kd = 0.35  # Freno aerodinámico
        
        # Límites de saturación
        self.max_vel = 0.5          # Velocidad máxima estricta (50 cm/s)
        self.max_integral = 1.0     # Límite de acumulación
        
        # Escala de velocidad (1.0 para Gazebo, 100.0 para Tello real)
        self.declare_parameter('velocity_scale', 1.0)
        self.vel_scale = self.get_parameter('velocity_scale').get_parameter_value().double_value
        
        # Memoria de estado para derivadas e integrales
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_z = 0.0
        
        # Filtro para la derivada
        self.filtered_dx = 0.0
        self.filtered_dy = 0.0
        self.filtered_dz = 0.0
        
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
        self.get_logger().info(f"Nuevo objetivo recibido: X={self.target_x}, Y={self.target_y}, Z={self.target_z}")
        
    def control_loop(self):
        if self.current_pose is None or not self.target_received:
            return
            
        # Seguridad: Esperar a que el dron termine el despegue físico (Z > 40 cm)
        # Esto evita enviar comandos agresivos si el objetivo se manda muy pronto.
        if self.current_pose.z < 0.4:
            self.get_logger().info("Esperando a que termine el despegue (Z < 0.4m)...", throttle_duration_sec=2.0)
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
        
        # Tolerancia para detenerse (15 cm para evitar salir y entrar por ruido)
        if distance > 0.15: 
            
            # 2. Calcular Integrales (I) y aplicar Anti-windup
            self.integral_x += error_x * dt
            self.integral_y += error_y * dt
            self.integral_z += error_z * dt
            
            self.integral_x = max(min(self.integral_x, self.max_integral), -self.max_integral)
            self.integral_y = max(min(self.integral_y, self.max_integral), -self.max_integral)
            self.integral_z = max(min(self.integral_z, self.max_integral), -self.max_integral)
            
            # 3. Calcular Derivadas (D)
            raw_dx = (error_x - self.prev_error_x) / dt
            raw_dy = (error_y - self.prev_error_y) / dt
            raw_dz = (error_z - self.prev_error_z) / dt
            
            # Suavizar derivada (EMA) para que el ruido de posición no haga picos locos
            alpha_d = 0.2
            self.filtered_dx = (alpha_d * raw_dx) + ((1.0 - alpha_d) * self.filtered_dx)
            self.filtered_dy = (alpha_d * raw_dy) + ((1.0 - alpha_d) * self.filtered_dy)
            self.filtered_dz = (alpha_d * raw_dz) + ((1.0 - alpha_d) * self.filtered_dz)
            
            # 4. Ecuación PID
            vel_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * self.filtered_dx)
            vel_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * self.filtered_dy)
            vel_z = (self.kp * error_z) + (self.ki * self.integral_z) + (self.kd * self.filtered_dz)
            
            # 5. Aplicar saturación (Clamp de velocidad)
            twist.linear.x = max(min(vel_x, self.max_vel), -self.max_vel)
            twist.linear.y = max(min(vel_y, self.max_vel), -self.max_vel)
            twist.linear.z = max(min(vel_z, self.max_vel), -self.max_vel)
            
        else:
            # Detener el dron al alcanzar el objetivo, pero NO resetear integrales.
            # Conservar las integrales permite que el dron siga haciendo fuerza
            # contra el viento o la deriva aerodinámica.
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            self.get_logger().info("Posición objetivo alcanzada de forma estable", throttle_duration_sec=2.0)
            
        # Guardar estado para la siguiente iteración
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_z = error_z
        self.last_time = current_time
            
        # Escalar velocidades para el driver si es necesario
        twist.linear.x *= self.vel_scale
        twist.linear.y *= self.vel_scale
        twist.linear.z *= self.vel_scale

        # Publicar el comando de velocidad
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TelloPositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Deteniendo controlador...")
    finally:
        stop_msg = Twist()
        node.cmd_vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()