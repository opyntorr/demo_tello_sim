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

        # ── Publicador ────────────────────────────────────────────────
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
        self.P = np.eye(7) * 0.1            # Covarianza inicial

        # ── Ruido de proceso (Q) ──────────────────────────────────────
        self.Q = np.diag([
            0.01,   # px  - poca incertidumbre en posición
            0.01,   # py
            0.01,   # pz
            0.5,    # vx  - más incertidumbre en velocidad
            0.5,    # vy
            0.5,    # vz
            0.01    # θ
        ])

        # ── Ruido de medición (R) por sensor ──────────────────────────
        self.R_vel = np.diag([0.3, 0.3, 0.3])    # Flujo óptico (ruidoso)
        self.R_tof = np.array([[0.05]])            # TOF (bastante preciso)
        self.R_yaw = np.array([[0.1]])             # IMU yaw

        # ── Matrices de observación (H) ───────────────────────────────
        # Velocidad: observamos [vx, vy, vz] del estado
        self.H_vel = np.zeros((3, 7))
        self.H_vel[0, 3] = 1.0  # vx
        self.H_vel[1, 4] = 1.0  # vy
        self.H_vel[2, 5] = 1.0  # vz

        # TOF: observamos pz del estado
        self.H_tof = np.zeros((1, 7))
        self.H_tof[0, 2] = 1.0  # pz

        # Yaw IMU: observamos θ del estado
        self.H_yaw = np.zeros((1, 7))
        self.H_yaw[0, 6] = 1.0  # θ

        # ── Variables auxiliares ───────────────────────────────────────
        self.omega_z = 0.0                   # Velocidad angular actual
        self.last_time = self.get_clock().now()
        self.imu_yaw = None                  # Último yaw del IMU

        # ── Control de TOF: solo durante takeoff ──────────────────────
        self.takeoff_complete = False         # Se activa al alcanzar altura
        self.declare_parameter('takeoff_height', 0.5)  # Umbral en metros
        self.takeoff_height = self.get_parameter(
            'takeoff_height'
        ).get_parameter_value().double_value

        # ── Bucle de predicción a 50 Hz ───────────────────────────────
        self.timer = self.create_timer(0.02, self.predict_loop)

        self.get_logger().info(
            f"EKF Odometry iniciado (vel_multiplier={self.multiplier}, "
            f"takeoff_height={self.takeoff_height}m)"
        )

    # ──────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ──────────────────────────────────────────────────────────────────

    def odom_callback(self, msg):
        """Recibe velocidades del flujo óptico y altura del TOF."""
        # 1. Leer velocidades en body frame y aplicar multiplicador
        vx_body = msg.twist.twist.linear.x * self.multiplier
        vy_body = msg.twist.twist.linear.y * self.multiplier
        vz      = msg.twist.twist.linear.z * self.multiplier

        # Guardamos omega_z para la predicción
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
        if not self.takeoff_complete:
            if 0.01 < incoming_z < 5.0:
                z_tof = np.array([incoming_z])
                self._ekf_update(z_tof, self.H_tof, self.R_tof)

                # Verificar si el takeoff se completó
                if incoming_z >= self.takeoff_height:
                    self.takeoff_complete = True
                    self.get_logger().info(
                        f"Takeoff completado (TOF={incoming_z:.2f}m). "
                        f"Cambiando a odometría pura para Z."
                    )

    def imu_callback(self, msg):
        """Recibe la orientación del IMU para corregir el yaw."""
        # Extraer yaw del quaternion
        q = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        _, _, yaw = tf_transformations.euler_from_quaternion(q)

        # Corrección EKF con medición de yaw
        z_yaw = np.array([yaw])
        self._ekf_update(z_yaw, self.H_yaw, self.R_yaw)

    # ──────────────────────────────────────────────────────────────────
    # EKF: PREDICCIÓN
    # ──────────────────────────────────────────────────────────────────

    def predict_loop(self):
        """Paso de predicción del EKF a 50 Hz."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        if dt <= 0.0:
            return

        # 1. Modelo de predicción f(x, u)
        px, py, pz, vx, vy, vz, theta = self.x

        self.x[0] = px + vx * dt       # px
        self.x[1] = py + vy * dt       # py
        self.x[2] = pz + vz * dt       # pz
        # vx, vy, vz se mantienen (modelo de velocidad constante)
        self.x[6] = theta + self.omega_z * dt  # θ

        # 2. Jacobiano F (derivada parcial de f respecto a x)
        F = np.eye(7)
        F[0, 3] = dt   # ∂px/∂vx
        F[1, 4] = dt   # ∂py/∂vy
        F[2, 5] = dt   # ∂pz/∂vz

        # 3. Propagar covarianza: P = F * P * Fᵀ + Q
        self.P = F @ self.P @ F.T + self.Q * dt

        # 4. Publicar la odometría estimada
        self._publish_odom(current_time)

        self.last_time = current_time

    # ──────────────────────────────────────────────────────────────────
    # EKF: CORRECCIÓN (UPDATE)
    # ──────────────────────────────────────────────────────────────────

    def _ekf_update(self, z, H, R):
        """
        Paso de corrección genérico del EKF.
        
        Args:
            z: Vector de medición (numpy array)
            H: Matriz de observación (numpy array)
            R: Matriz de covarianza de ruido de medición (numpy array)
        """
        # 1. Innovación: y = z - H * x
        y = z - H @ self.x

        # 2. Covarianza de innovación: S = H * P * Hᵀ + R
        S = H @ self.P @ H.T + R

        # 3. Ganancia de Kalman: K = P * Hᵀ * S⁻¹
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4. Actualizar estado: x = x + K * y
        self.x = self.x + K @ y

        # 5. Actualizar covarianza: P = (I - K * H) * P
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

        # Posición estimada
        odom_msg.pose.pose.position.x = self.x[0]
        odom_msg.pose.pose.position.y = self.x[1]
        odom_msg.pose.pose.position.z = self.x[2]

        # Orientación (θ → quaternion)
        q = tf_transformations.quaternion_from_euler(0, 0, self.x[6])
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        # Velocidades estimadas (en world frame)
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
