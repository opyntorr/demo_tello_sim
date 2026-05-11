import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
import math
from rclpy.signals import SignalHandlerOptions

class TelloPositionController(Node):
    def __init__(self):
        super().__init__('tello_position_controller')

        # Publicadores y Suscriptores
        self.cmd_vel_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.target_sub = self.create_subscription(Point, '/drone1/target_position', self.target_callback, 10)
        self.land_sub = self.create_subscription(Empty, '/land', self._safety_stop, 1)
        self.emergency_sub = self.create_subscription(Empty, '/emergency', self._safety_stop, 1)

        self.active = True
        
        # Posición objetivo inicializada en None (esperando comando)
        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.target_received = False
        self.has_taken_off = False
        self.takeoff_complete_time = None
        
        # Ganancias del controlador PID (Configurables)
        self.declare_parameter('kp', 2.5)
        self.declare_parameter('ki', 0.8)
        self.declare_parameter('kd', 0.0)
        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.ki = self.get_parameter('ki').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value
        
        # Límites de saturación
        self.max_vel = 0.6          # Velocidad máxima estricta (50 cm/s)
        self.max_integral = 1.0     # Límite de acumulación
        
        # Escala de velocidad (1.0 para Gazebo, 100.0 para Tello real)
        self.declare_parameter('velocity_scale', 1.0)
        self.vel_scale = self.get_parameter('velocity_scale').get_parameter_value().double_value
        
        # --- NUEVOS PARÁMETROS DE SEGURIDAD ---
        self.declare_parameter('max_altitude', 3.0)     # Altura máxima permitida
        self.declare_parameter('min_altitude', 0.2)     # Altura mínima para considerar vuelo
        self.declare_parameter('max_xy_radius', 4.0)    # Radio máximo desde el origen (0,0)
        self.declare_parameter('odom_timeout', 0.5)     # Tiempo máximo sin odometría (segundos)
        
        self.max_alt = self.get_parameter('max_altitude').get_parameter_value().double_value
        self.min_alt = self.get_parameter('min_altitude').get_parameter_value().double_value
        self.max_xy = self.get_parameter('max_xy_radius').get_parameter_value().double_value
        self.odom_timeout = self.get_parameter('odom_timeout').get_parameter_value().double_value
        

        
        # Memoria de estado para derivadas e integrales
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_z = 0.0
        
        # Filtro para la derivada (EMA)
        self.filtered_dx = 0.0
        self.filtered_dy = 0.0
        self.filtered_dz = 0.0
        
        self.current_pose = None
        self.current_orientation = None
        self.last_time = None
        self.last_odom_time = self.get_clock().now()
        self.start_time = None
        
        # Bucle de control a 100 Hz (sincronizado con OptiTrack a 120 Hz)
        self.timer = self.create_timer(1.0/100.0, self.control_loop)
        
    def _safety_stop(self, msg):
        self.active = False
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().warn("Land/Emergency recibido — controlador silenciado")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation
        self.last_odom_time = self.get_clock().now()
        
        
    def target_callback(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y
        self.target_z = msg.z
        self.target_received = True
        self.get_logger().info(f"Nuevo objetivo recibido: X={self.target_x}, Y={self.target_y}, Z={self.target_z}")
        
    def control_loop(self):
        if not self.active:
            return

        current_time = self.get_clock().now()
        
        # 1. SEGURIDAD: Verificar Watchdog de Odometría (Pérdida de localización)
        dt_odom = (current_time - self.last_odom_time).nanoseconds / 1e9
        if dt_odom > self.odom_timeout:
            self.get_logger().error(f"¡LOCALIZACIÓN PERDIDA! Hace {dt_odom:.2f}s. Deteniendo motores.", throttle_duration_sec=1.0)
            self.cmd_vel_pub.publish(Twist())
            return

        if self.current_pose is None:
            return
            
        # 2. SEGURIDAD: Geofencing (Límites de espacio)
        # Altura máxima
        if self.current_pose.z > self.max_alt:
            self.get_logger().warn(f"¡Límite de altura excedido! ({self.current_pose.z:.2f}m > {self.max_alt}m)", throttle_duration_sec=1.0)
            # Forzar aterrizaje o comando de descenso agresivo
            twist = Twist()
            twist.linear.z = -0.5 * self.vel_scale
            self.cmd_vel_pub.publish(twist)
            return

        # Radio XY
        dist_xy = math.sqrt(self.current_pose.x**2 + self.current_pose.y**2)
        if dist_xy > self.max_xy:
            self.get_logger().error(f"¡FUERA DE RANGO XY! ({dist_xy:.2f}m > {self.max_xy}m). Regresando...", throttle_duration_sec=1.0)
            # Aquí podríamos implementar un "Return to Home", por ahora solo detenemos
            self.cmd_vel_pub.publish(Twist())
            return
            
        # Seguridad: Esperar a que el dron termine el despegue físico
        if not self.has_taken_off:
            if self.current_pose.z < 0.8:
                self.get_logger().info("Esperando a que termine el despegue (Z < 0.8m)...", throttle_duration_sec=2.0)
                self.cmd_vel_pub.publish(Twist()) # Enviar comandos 0 para que el drift actúe
                return
            else:
                self.has_taken_off = True
                self.takeoff_complete_time = current_time
                self.get_logger().info("¡Despegue detectado! Esperando estabilización macro (1.5s).")
                self.cmd_vel_pub.publish(Twist())
                return
                
        # Fase de estabilización: Permitir que el Tello termine su macro interno
        dt_since_takeoff = (current_time - self.takeoff_complete_time).nanoseconds / 1e9
        if dt_since_takeoff < 1.5:
            self.get_logger().info(f"Estabilizando despegue ({dt_since_takeoff:.1f}/1.5s)...", throttle_duration_sec=0.5)
            self.cmd_vel_pub.publish(Twist())
            return
            
        # Capturar la posición actual como meta (Hover automático) si aún no hemos recibido un objetivo manual
        if not self.target_received:
            self.target_x = self.current_pose.x
            self.target_y = self.current_pose.y
            self.target_z = 2.5
            self.target_received = True
            self.get_logger().info(f"¡Hover automático activado tras deriva! Anclando a: X={self.target_x:.2f}, Y={self.target_y:.2f}, Z={self.target_z:.2f}")
            
        # Inicializar el tiempo en la primera ejecución activa
        if self.last_time is None:
            self.last_time = current_time
            return
            
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
            
            # Suavizar derivada (EMA) — alpha bajo para absorber spikes de sensores reales
            alpha_d = 0.05
            self.filtered_dx = (alpha_d * raw_dx) + ((1.0 - alpha_d) * self.filtered_dx)
            self.filtered_dy = (alpha_d * raw_dy) + ((1.0 - alpha_d) * self.filtered_dy)
            self.filtered_dz = (alpha_d * raw_dz) + ((1.0 - alpha_d) * self.filtered_dz)
            
            # 4. Ecuación PID (usando derivadas filtradas - cálculo en el marco global de OptiTrack)
            global_vel_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * self.filtered_dx)
            global_vel_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * self.filtered_dy)
            vel_z = (self.kp * error_z) + (self.ki * self.integral_z) + (self.kd * self.filtered_dz)
            
            # --- ROTACIÓN DE MARCOS: DEL GLOBAL AL LOCAL ---
            # Extraer el ángulo Yaw (Z) del dron a partir del cuaternión de odometría
            q = self.current_orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Rotar el vector de velocidad global para que tenga sentido desde el punto de vista del dron
            local_vel_x = global_vel_x * math.cos(yaw) + global_vel_y * math.sin(yaw)
            local_vel_y = -global_vel_x * math.sin(yaw) + global_vel_y * math.cos(yaw)
            
            # 5. Aplicar saturación (Clamp de velocidad en el marco local)
            twist.linear.x = max(min(local_vel_x, self.max_vel), -self.max_vel)
            twist.linear.y = max(min(local_vel_y, self.max_vel), -self.max_vel)
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