#!/usr/bin/env python3
"""
Nodo de Odometría con Filtro de Kalman Extendido (EKF).

Fusiona tres fuentes de datos para estimar la posición del dron:
  1. Velocidades del flujo óptico (body frame) → /drone1/odom
  2. Altura absoluta del sensor TOF              → /drone1/odom (solo durante takeoff)
  3. Orientación (yaw) del IMU                   → /drone1/imu

Vector de estado (7 elementos):
  x = [px, py, pz, vx, vy, vz, θ]ᵀ

El TOF solo se utiliza para establecer la altura inicial durante el
despegue. Una vez completado, la altura se estima únicamente por
integración de velocidad (odometría).

Publica la odometría filtrada en /drone1/integrated_odom,
manteniendo la misma interfaz que el integrador simple anterior.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64
import numpy as np
import math

# Parche para compatibilidad con Numpy 1.24+
if not hasattr(np, "float"):
    np.float = float

import tf_transformations


class EKFOdometryNode(Node):
    def __init__(self):
        super().__init__('simple_integrator_odom')

        # ── Suscriptores ──────────────────────────────────────────────
        self.vel_sub = self.create_subscription(
            Odometry, '/drone1/odom',
            self.odom_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/drone1/imu',
            self.imu_callback, 10
        )

        # ── Publicador principal ──────────────────────────────────────
        self.odom_pub = self.create_publisher(
            Odometry, '/drone1/integrated_odom', 10
        )

        # ── Parámetro de escala de velocidad ──────────────────────────
        self.declare_parameter('vel_multiplier', 10.0)
        self.multiplier = self.get_parameter(
            'vel_multiplier'
        ).get_parameter_value().double_value

        # ── Estado del EKF ────────────────────────────────────────────
        # x = [px, py, pz, vx, vy, vz, θ]
        self.x = np.zeros(7)
        self.P = np.eye(7) * 0.1

        # ── Ruido de proceso (Q) ──────────────────────────────────────
        self.Q = np.diag([
            0.01,   # px
            0.01,   # py
            0.01,   # pz
            0.5,    # vx
            0.5,    # vy
            0.5,    # vz
            0.01    # θ
        ])

        # ── Ruido de medición (R) por sensor ──────────────────────────
        self.R_vel = np.diag([0.3, 0.3, 0.3])
        self.R_tof = np.array([[0.05]])
        self.R_yaw = np.array([[0.1]])

        # ── Matrices de observación (H) ───────────────────────────────
        self.H_vel = np.zeros((3, 7))
        self.H_vel[0, 3] = 1.0
        self.H_vel[1, 4] = 1.0
        self.H_vel[2, 5] = 1.0

        self.H_tof = np.zeros((1, 7))
        self.H_tof[0, 2] = 1.0

        self.H_yaw = np.zeros((1, 7))
        self.H_yaw[0, 6] = 1.0

        # ── Variables auxiliares ───────────────────────────────────────
        self.omega_z = 0.0
        self.last_time = self.get_clock().now()
        self.imu_yaw = None

        # ── Control de TOF: solo durante takeoff ──────────────────────
        self.takeoff_complete = False
        self.declare_parameter('takeoff_height', 0.5)
        self.takeoff_height = self.get_parameter(
            'takeoff_height'
        ).get_parameter_value().double_value

        # ── Publicadores de diagnóstico (PlotJuggler) ─────────────────
        # Estado EKF
        self._dbg_pose_pub    = self.create_publisher(Vector3Stamped, '/debug/ekf/pose',     10)
        self._dbg_vel_pub     = self.create_publisher(Vector3Stamped, '/debug/ekf/velocity', 10)
        self._dbg_yaw_pub     = self.create_publisher(Float64,        '/debug/ekf/yaw',      10)
        # Incertidumbre del filtro (diagonal de P)
        self._dbg_cov_pos_pub = self.create_publisher(Vector3Stamped, '/debug/ekf/cov_pos',  10)
        self._dbg_cov_vel_pub = self.create_publisher(Vector3Stamped, '/debug/ekf/cov_vel',  10)
        # Señales de los sensores de entrada
        self._dbg_odom_vel_pub  = self.create_publisher(Vector3Stamped, '/debug/ekf/odom_vel_raw', 10)
        self._dbg_tof_pub       = self.create_publisher(Float64,        '/debug/ekf/tof_z',        10)
        self._dbg_tof_valid_pub = self.create_publisher(Float64,        '/debug/ekf/tof_valid',    10)
        self._dbg_takeoff_pub   = self.create_publisher(Float64,        '/debug/ekf/takeoff_done', 10)

        # ── Watchdog ──────────────────────────────────────────────────
        self._last_odom_time = 0.0
        self._last_imu_time  = 0.0
        self.watchdog_timer = self.create_timer(1.0, self._watchdog)

        # ── Bucle de predicción a 50 Hz ───────────────────────────────
        self.timer = self.create_timer(0.02, self.predict_loop)

        self.get_logger().info(
            f'[ekf] Nodo iniciado | vel_multiplier={self.multiplier}'
            f' takeoff_height={self.takeoff_height} m'
        )

    # ──────────────────────────────────────────────────────────────────
    # DIAGNÓSTICO
    # ──────────────────────────────────────────────────────────────────

    def _pub_v3(self, pub, x, y, z, stamp):
        """Publish a Vector3Stamped with a shared stamp."""
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.vector.x = float(x)
        msg.vector.y = float(y)
        msg.vector.z = float(z)
        pub.publish(msg)

    def _publish_debug(self, stamp):
        """
        Publish all EKF diagnostic topics.

        Llamado en cada paso de predicción (50 Hz) para que PlotJuggler
        reciba datos continuos independientemente de la tasa de los sensores.
        """
        now_stamp = stamp.to_msg()

        # Estado estimado
        self._pub_v3(self._dbg_pose_pub, self.x[0], self.x[1], self.x[2], now_stamp)
        self._pub_v3(self._dbg_vel_pub,  self.x[3], self.x[4], self.x[5], now_stamp)

        yaw_msg = Float64()
        yaw_msg.data = float(self.x[6])
        self._dbg_yaw_pub.publish(yaw_msg)

        # Incertidumbre (diagonal de la covarianza)
        self._pub_v3(self._dbg_cov_pos_pub,
                     self.P[0, 0], self.P[1, 1], self.P[2, 2], now_stamp)
        self._pub_v3(self._dbg_cov_vel_pub,
                     self.P[3, 3], self.P[4, 4], self.P[5, 5], now_stamp)

        # Estado de despegue
        tkoff_msg = Float64()
        tkoff_msg.data = 1.0 if self.takeoff_complete else 0.0
        self._dbg_takeoff_pub.publish(tkoff_msg)

    def _watchdog(self):
        """Advierte cuando algún sensor lleva demasiado tiempo sin publicar."""
        now = self.get_clock().now().nanoseconds / 1e9
        if self._last_odom_time > 0.0 and now - self._last_odom_time > 0.5:
            self.get_logger().warn(
                f'[ekf] ADVERTENCIA: sin datos de /drone1/odom'
                f' en {now - self._last_odom_time:.2f} s'
            )
        if self._last_imu_time > 0.0 and now - self._last_imu_time > 0.5:
            self.get_logger().warn(
                f'[ekf] ADVERTENCIA: sin datos de /drone1/imu'
                f' en {now - self._last_imu_time:.2f} s'
            )

    # ──────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ──────────────────────────────────────────────────────────────────

    def odom_callback(self, msg):
        """Recibe velocidades del flujo óptico y altura del TOF."""
        self._last_odom_time = self.get_clock().now().nanoseconds / 1e9

        # 1. Leer velocidades en body frame y aplicar multiplicador
        vx_body = msg.twist.twist.linear.x * self.multiplier
        vy_body = msg.twist.twist.linear.y * self.multiplier
        vz      = msg.twist.twist.linear.z * self.multiplier

        self.omega_z = msg.twist.twist.angular.z

        # 2. Rotar velocidad de body → world usando el θ estimado
        theta = self.x[6]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        vx_world = vx_body * cos_t - vy_body * sin_t
        vy_world = vx_body * sin_t + vy_body * cos_t

        # 3. Corrección EKF con medición de velocidad
        z_vel = np.array([vx_world, vy_world, vz])
        self._ekf_update(z_vel, self.H_vel, self.R_vel)

        # 4. Corrección EKF con medición de TOF (solo durante takeoff)
        incoming_z = msg.pose.pose.position.z
        tof_valid = 0.01 < incoming_z < 5.0

        if not self.takeoff_complete and tof_valid:
            z_tof = np.array([incoming_z])
            self._ekf_update(z_tof, self.H_tof, self.R_tof)

            if incoming_z >= self.takeoff_height:
                self.takeoff_complete = True
                self.get_logger().info(
                    f'[ekf] Takeoff completado (TOF={incoming_z:.2f} m).'
                    f' Cambiando a odometría pura para Z.'
                )

        # Diagnóstico de sensores de entrada
        now_stamp = self.get_clock().now().to_msg()

        self._pub_v3(self._dbg_odom_vel_pub, vx_body, vy_body, vz, now_stamp)

        tof_msg = Float64()
        tof_msg.data = float(incoming_z)
        self._dbg_tof_pub.publish(tof_msg)

        valid_msg = Float64()
        valid_msg.data = 1.0 if tof_valid else 0.0
        self._dbg_tof_valid_pub.publish(valid_msg)

        self.get_logger().info(
            f'[ekf] odom: vel_body=({vx_body:.3f},{vy_body:.3f},{vz:.3f})'
            f' tof_z={incoming_z:.3f} [{"VÁLIDO" if tof_valid else "INVÁLIDO"}]',
            throttle_duration_sec=2.0,
        )

    def imu_callback(self, msg):
        """Recibe la orientación del IMU para corregir el yaw."""
        self._last_imu_time = self.get_clock().now().nanoseconds / 1e9

        q = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        _, _, yaw = tf_transformations.euler_from_quaternion(q)

        z_yaw = np.array([yaw])
        self._ekf_update(z_yaw, self.H_yaw, self.R_yaw)

        self.get_logger().info(
            f'[ekf] imu: yaw={yaw:.3f} rad  yaw_estimado={self.x[6]:.3f} rad',
            throttle_duration_sec=2.0,
        )

    # ──────────────────────────────────────────────────────────────────
    # EKF: PREDICCIÓN
    # ──────────────────────────────────────────────────────────────────

    def predict_loop(self):
        """Paso de predicción del EKF a 50 Hz."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        if dt <= 0.0:
            return

        if dt > 1.0:
            self.get_logger().warn(f'[ekf] ADVERTENCIA: dt anómalo ({dt:.4f} s)')

        # 1. Modelo de predicción f(x, u)
        px, py, pz, vx, vy, vz, theta = self.x

        self.x[0] = px + vx * dt
        self.x[1] = py + vy * dt
        self.x[2] = pz + vz * dt
        self.x[6] = theta + self.omega_z * dt

        # 2. Jacobiano F
        F = np.eye(7)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # 3. Propagar covarianza: P = F * P * Fᵀ + Q
        self.P = F @ self.P @ F.T + self.Q * dt

        # 4. Publicar odometría estimada + diagnósticos
        self._publish_odom(current_time)
        self._publish_debug(current_time)

        self.last_time = current_time

        self.get_logger().info(
            f'[ekf] estado: pos=({self.x[0]:.3f},{self.x[1]:.3f},{self.x[2]:.3f})'
            f' vel=({self.x[3]:.3f},{self.x[4]:.3f},{self.x[5]:.3f})'
            f' θ={self.x[6]:.3f} rad'
            f' | cov_pos=({self.P[0,0]:.4f},{self.P[1,1]:.4f},{self.P[2,2]:.4f})',
            throttle_duration_sec=2.0,
        )

    # ──────────────────────────────────────────────────────────────────
    # EKF: CORRECCIÓN (UPDATE)
    # ──────────────────────────────────────────────────────────────────

    def _ekf_update(self, z, H, R):
        """Ejecuta el paso de corrección genérico del EKF."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(7)
        self.P = (I - K @ H) @ self.P

    # ──────────────────────────────────────────────────────────────────
    # PUBLICACIÓN
    # ──────────────────────────────────────────────────────────────────

    def _publish_odom(self, stamp):
        """Publica la odometría estimada por el EKF."""
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = self.x[0]
        odom_msg.pose.pose.position.y = self.x[1]
        odom_msg.pose.pose.position.z = self.x[2]

        q = tf_transformations.quaternion_from_euler(0, 0, self.x[6])
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        odom_msg.twist.twist.linear.x = self.x[3]
        odom_msg.twist.twist.linear.y = self.x[4]
        odom_msg.twist.twist.linear.z = self.x[5]
        odom_msg.twist.twist.angular.z = self.omega_z

        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
