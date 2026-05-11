import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Vector3Stamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import math
from rclpy.signals import SignalHandlerOptions


class TelloPositionController(Node):
    def __init__(self):
        super().__init__('tello_position_controller')

        # Publicadores y Suscriptores
        self.cmd_vel_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/drone1/integrated_odom', self.odom_callback, 10
        )
        self.target_sub = self.create_subscription(
            Point, '/drone1/target_position', self.target_callback, 10
        )

        # Publicadores de diagnóstico — PlotJuggler (Vector3Stamped lleva header.stamp
        # para alinear correctamente trazas de distintas frecuencias en la línea de tiempo)
        self._dbg_err_pub  = self.create_publisher(Vector3Stamped, '/debug/ctrl/error',    10)
        self._dbg_p_pub    = self.create_publisher(Vector3Stamped, '/debug/ctrl/pid_p',    10)
        self._dbg_i_pub    = self.create_publisher(Vector3Stamped, '/debug/ctrl/pid_i',    10)
        self._dbg_d_pub    = self.create_publisher(Vector3Stamped, '/debug/ctrl/pid_d',    10)
        self._dbg_raw_pub  = self.create_publisher(Vector3Stamped, '/debug/ctrl/cmd_raw',  10)
        self._dbg_sent_pub = self.create_publisher(Vector3Stamped, '/debug/ctrl/cmd_sent', 10)
        self._dbg_dist_pub = self.create_publisher(Float64,        '/debug/ctrl/distance', 10)
        self._dbg_dt_pub   = self.create_publisher(Float64,        '/debug/ctrl/dt',       10)

        # Posición objetivo inicializada en None (esperando comando)
        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.target_received = False

        # Ganancias del controlador PID
        self.kp = 0.3
        self.ki = 0.01
        self.kd = 0.5

        # Límites de saturación
        self.max_vel = 0.5
        self.max_integral = 1.0

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

        self.current_pose = None
        self.last_time = None
        self.start_time = None

        # Watchdog: tiempo de la última odometría integrada recibida
        self._last_integrated_time = 0.0
        self.watchdog_timer = self.create_timer(1.0, self._watchdog)

        # Bucle de control a 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

    # ------------------------------------------------------------------

    def _pub_v3(self, pub, x, y, z, stamp):
        """
        Publish a Vector3Stamped diagnostic message.

        Reutiliza el stamp ya calculado en el tick de control para que
        todos los tópicos de diagnóstico queden alineados en PlotJuggler.
        """
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.vector.x = float(x)
        msg.vector.y = float(y)
        msg.vector.z = float(z)
        pub.publish(msg)

    def _watchdog(self):
        """Advierte cuando /drone1/integrated_odom lleva demasiado tiempo sin publicar."""
        now = self.get_clock().now().nanoseconds / 1e9
        if self._last_integrated_time > 0.0 and now - self._last_integrated_time > 0.5:
            self.get_logger().warn(
                f'[ctrl] ADVERTENCIA: sin datos de /drone1/integrated_odom'
                f' en {now - self._last_integrated_time:.2f} s'
            )

    # ------------------------------------------------------------------

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position
        self._last_integrated_time = self.get_clock().now().nanoseconds / 1e9

    def target_callback(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y
        self.target_z = msg.z
        self.target_received = True
        self.get_logger().info(
            f'Nuevo objetivo recibido: X={self.target_x}, Y={self.target_y}, Z={self.target_z}'
        )

    def control_loop(self):
        if self.current_pose is None or not self.target_received:
            return

        # Seguridad: esperar a que el dron termine el despegue físico (Z > 40 cm)
        if self.current_pose.z < 0.4:
            self.get_logger().warn(
                f'[ctrl] ADVERTENCIA: esperando despegue'
                f' (Z={self.current_pose.z:.2f} m < 0.4 m)',
                throttle_duration_sec=2.0,
            )
            return

        current_time = self.get_clock().now()

        if self.last_time is None:
            self.last_time = current_time
            self.start_time = current_time
            return

        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        if dt > 1.0:
            self.get_logger().warn(f'[ctrl] ADVERTENCIA: dt anómalo ({dt:.4f} s)')

        # 1. Errores actuales
        error_x = self.target_x - self.current_pose.x
        error_y = self.target_y - self.current_pose.y
        error_z = self.target_z - self.current_pose.z

        distance = math.sqrt(error_x**2 + error_y**2 + error_z**2)

        # Valores PID por defecto (cero cuando el objetivo ya está alcanzado)
        p_x = p_y = p_z = 0.0
        i_x = i_y = i_z = 0.0
        d_x = d_y = d_z = 0.0
        vel_x = vel_y = vel_z = 0.0

        twist = Twist()

        if distance > 0.10:
            # 2. Integrales con anti-windup
            self.integral_x += error_x * dt
            self.integral_y += error_y * dt
            self.integral_z += error_z * dt

            self.integral_x = max(min(self.integral_x, self.max_integral), -self.max_integral)
            self.integral_y = max(min(self.integral_y, self.max_integral), -self.max_integral)
            self.integral_z = max(min(self.integral_z, self.max_integral), -self.max_integral)

            # Avisar si la integral se acerca al límite de anti-windup
            windup_threshold = self.max_integral * 0.95
            for axis, val in [('X', self.integral_x), ('Y', self.integral_y), ('Z', self.integral_z)]:
                if abs(val) >= windup_threshold:
                    self.get_logger().warn(
                        f'[ctrl] ADVERTENCIA: integral en eje {axis} saturada ({val:.3f})'
                    )

            # 3. Derivadas
            derivative_x = (error_x - self.prev_error_x) / dt
            derivative_y = (error_y - self.prev_error_y) / dt
            derivative_z = (error_z - self.prev_error_z) / dt

            # 4. Términos PID separados para diagnóstico
            p_x = self.kp * error_x
            p_y = self.kp * error_y
            p_z = self.kp * error_z

            i_x = self.ki * self.integral_x
            i_y = self.ki * self.integral_y
            i_z = self.ki * self.integral_z

            d_x = self.kd * derivative_x
            d_y = self.kd * derivative_y
            d_z = self.kd * derivative_z

            vel_x = p_x + i_x + d_x
            vel_y = p_y + i_y + d_y
            vel_z = p_z + i_z + d_z

            # 5. Saturación de velocidad
            twist.linear.x = max(min(vel_x, self.max_vel), -self.max_vel)
            twist.linear.y = max(min(vel_y, self.max_vel), -self.max_vel)
            twist.linear.z = max(min(vel_z, self.max_vel), -self.max_vel)

        else:
            # Detener el dron y resetear integrales al alcanzar el objetivo
            self.integral_x = 0.0
            self.integral_y = 0.0
            self.integral_z = 0.0
            self.get_logger().info(
                'Posición objetivo alcanzada de forma estable', throttle_duration_sec=2.0
            )

        # Guardar estado para la siguiente iteración
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_z = error_z
        self.last_time = current_time

        # Escalar velocidades para el driver si es necesario
        twist.linear.x *= self.vel_scale
        twist.linear.y *= self.vel_scale
        twist.linear.z *= self.vel_scale

        self.cmd_vel_pub.publish(twist)

        # --- Publicar diagnósticos (PlotJuggler) ---
        now_stamp = self.get_clock().now().to_msg()

        self._pub_v3(self._dbg_err_pub,  error_x, error_y, error_z, now_stamp)
        self._pub_v3(self._dbg_p_pub,    p_x,     p_y,     p_z,     now_stamp)
        self._pub_v3(self._dbg_i_pub,    i_x,     i_y,     i_z,     now_stamp)
        self._pub_v3(self._dbg_d_pub,    d_x,     d_y,     d_z,     now_stamp)
        self._pub_v3(self._dbg_raw_pub,  vel_x,   vel_y,   vel_z,   now_stamp)
        # cmd_sent en m/s físicos (sin factor de escala del driver)
        self._pub_v3(
            self._dbg_sent_pub,
            twist.linear.x / self.vel_scale,
            twist.linear.y / self.vel_scale,
            twist.linear.z / self.vel_scale,
            now_stamp,
        )

        dist_msg = Float64()
        dist_msg.data = distance
        self._dbg_dist_pub.publish(dist_msg)

        dt_msg = Float64()
        dt_msg.data = dt
        self._dbg_dt_pub.publish(dt_msg)

        # --- Logs de texto (throttled 1 s) ---
        self.get_logger().info(
            f'[ctrl] pos=({self.current_pose.x:.3f},{self.current_pose.y:.3f},{self.current_pose.z:.3f})'
            f' objetivo=({self.target_x:.3f},{self.target_y:.3f},{self.target_z:.3f})'
            f' distancia={distance:.3f} m',
            throttle_duration_sec=1.0,
        )
        self.get_logger().info(
            f'[ctrl] dt={dt:.3f} s'
            f' | P=({p_x:.3f},{p_y:.3f},{p_z:.3f})'
            f' I=({i_x:.3f},{i_y:.3f},{i_z:.3f})'
            f' D=({d_x:.3f},{d_y:.3f},{d_z:.3f})'
            f' | vel_bruta=({vel_x:.3f},{vel_y:.3f},{vel_z:.3f})'
            f' | cmd=({twist.linear.x:.3f},{twist.linear.y:.3f},{twist.linear.z:.3f})',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TelloPositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Deteniendo controlador...')
    finally:
        stop_msg = Twist()
        node.cmd_vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
