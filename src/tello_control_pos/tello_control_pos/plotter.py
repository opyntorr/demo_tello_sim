import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import os
from rclpy.signals import SignalHandlerOptions

class TelloPlotter(Node):
    def __init__(self):
        super().__init__('tello_plotter')
        
        # Suscriptor a la odometría y a la meta
        self.odom_sub = self.create_subscription(Odometry, '/drone1/integrated_odom', self.odom_callback, 10)
        self.target_sub = self.create_subscription(Point, '/drone1/target_position', self.target_callback, 10)
        
        # Posición objetivo inicial
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_received = False
        
        # Historial de datos
        self.time_history = []
        self.x_history = []
        self.y_history = []
        self.z_history = []
        self.tx_history = []
        self.ty_history = []
        self.tz_history = []
        self.start_time = None
        
        # Configuración de gráfica en tiempo real
        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(3, 2, figure=self.fig)
        
        # Subplot 3D
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.ax3d.set_title('Trayectoria 3D (Tiempo Real)')
        self.ax3d.set_xlabel('X [m]')
        self.ax3d.set_ylabel('Y [m]')
        self.ax3d.set_zlabel('Z [m]')
        self.ax3d.plot([0], [0], [0], 'go', label='Inicio')
        
        self.target_line3d, = self.ax3d.plot([], [], [], 'r--', label='Meta')
        
        self.line3d, = self.ax3d.plot([], [], [], 'b-', label='Trayecto')
        
        # Límites fijos para la gráfica 3D
        self.ax3d.set_xlim([-0.5, max(3.0, self.target_x + 1)])
        self.ax3d.set_ylim([-0.5, max(3.0, self.target_y + 1)])
        self.ax3d.set_zlim([0, max(2.0, self.target_z + 1)])
        self.ax3d.legend()
        
        # Subplots X, Y, Z
        self.ax_x = self.fig.add_subplot(gs[0, 1])
        self.line_x, = self.ax_x.plot([], [], 'b-')
        self.target_line_x, = self.ax_x.plot([], [], 'r--', label='Meta X')
        self.ax_x.set_ylabel('X [m]')
        self.ax_x.grid(True)
        
        self.ax_y = self.fig.add_subplot(gs[1, 1])
        self.line_y, = self.ax_y.plot([], [], 'g-')
        self.target_line_y, = self.ax_y.plot([], [], 'r--', label='Meta Y')
        self.ax_y.set_ylabel('Y [m]')
        self.ax_y.grid(True)
        
        self.ax_z = self.fig.add_subplot(gs[2, 1])
        self.line_z, = self.ax_z.plot([], [], 'm-')
        self.target_line_z, = self.ax_z.plot([], [], 'r--', label='Meta Z')
        self.ax_z.set_ylabel('Z [m]')
        self.ax_z.set_xlabel('Tiempo [s]')
        self.ax_z.grid(True)
        
        self.fig.tight_layout()
        plt.show()

        # Timer para actualizar la gráfica (10 Hz)
        self.timer = self.create_timer(0.1, self.update_plot)
        self.messages_received = 0

    def odom_callback(self, msg):
        current_time = self.get_clock().now()
        
        if self.start_time is None:
            self.start_time = current_time
            
        t = (current_time - self.start_time).nanoseconds / 1e9
        pose = msg.pose.pose.position
        
        if not self.target_received:
            self.target_x = pose.x
            self.target_y = pose.y
            self.target_z = pose.z
            
        self.time_history.append(t)
        self.x_history.append(pose.x)
        self.y_history.append(pose.y)
        self.z_history.append(pose.z)
        self.tx_history.append(self.target_x)
        self.ty_history.append(self.target_y)
        self.tz_history.append(self.target_z)
        self.messages_received += 1

    def target_callback(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y
        self.target_z = msg.z
        self.target_received = True
        
        # Ajustar límites 3D si la meta está fuera de la vista
        curr_xlim = self.ax3d.get_xlim()
        curr_ylim = self.ax3d.get_ylim()
        curr_zlim = self.ax3d.get_zlim()
        self.ax3d.set_xlim([min(curr_xlim[0], -0.5), max(curr_xlim[1], self.target_x + 1)])
        self.ax3d.set_ylim([min(curr_ylim[0], -0.5), max(curr_ylim[1], self.target_y + 1)])
        self.ax3d.set_zlim([min(curr_zlim[0], 0), max(curr_zlim[1], self.target_z + 1)])

    def update_plot(self):
        if not self.time_history:
            return
            
        # Actualizar gráfica en tiempo real (cada 2 mensajes recibidos para no saturar)
        if self.messages_received % 2 == 0:
            try:
                self.line3d.set_data(self.x_history, self.y_history)
                self.line3d.set_3d_properties(self.z_history)
                
                self.target_line3d.set_data(self.tx_history, self.ty_history)
                self.target_line3d.set_3d_properties(self.tz_history)
                
                self.line_x.set_data(self.time_history, self.x_history)
                self.line_y.set_data(self.time_history, self.y_history)
                self.line_z.set_data(self.time_history, self.z_history)
                
                self.target_line_x.set_data(self.time_history, self.tx_history)
                self.target_line_y.set_data(self.time_history, self.ty_history)
                self.target_line_z.set_data(self.time_history, self.tz_history)
                
                max_t = max(5.0, self.time_history[-1])
                self.ax_x.set_xlim([0, max_t])
                self.ax_y.set_xlim([0, max_t])
                self.ax_z.set_xlim([0, max_t])
                
                min_x, max_x = min(self.x_history + [0, self.target_x]), max(self.x_history + [0, self.target_x])
                self.ax_x.set_ylim([min_x - 0.5, max_x + 0.5])
                
                min_y, max_y = min(self.y_history + [0, self.target_y]), max(self.y_history + [0, self.target_y])
                self.ax_y.set_ylim([min_y - 0.5, max_y + 0.5])
                
                min_z, max_z = min(self.z_history + [0, self.target_z]), max(self.z_history + [0, self.target_z])
                self.ax_z.set_ylim([min_z - 0.5, max_z + 0.5])
                
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception as e:
                pass

    def exportar_datos(self):
        dir_path = "/ros2_ws/src/tello_control_pos/tello_control_pos/"
        
        # 1. Guardar CSV fijo
        csv_file = os.path.join(dir_path, "ultimo_reporte_tello.csv")
        with open(csv_file, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "X", "Y", "Z", "Target_X", "Target_Y", "Target_Z"])
            for row in zip(self.time_history, self.x_history, self.y_history, self.z_history, self.tx_history, self.ty_history, self.tz_history): 
                writer.writerow(row)

        # 2. Guardar PNG de Posiciones X, Y, Z vs Tiempo
        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(self.time_history, self.x_history, 'b-', label='X')
        plt.plot(self.time_history, self.tx_history, 'r--', label='Meta X')
        plt.title('Posición en el tiempo')
        plt.ylabel('X [m]'); plt.grid(True); plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(self.time_history, self.y_history, 'g-', label='Y')
        plt.plot(self.time_history, self.ty_history, 'r--', label='Meta Y')
        plt.ylabel('Y [m]'); plt.grid(True); plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(self.time_history, self.z_history, 'm-', label='Z')
        plt.plot(self.time_history, self.tz_history, 'r--', label='Meta Z')
        plt.xlabel('Tiempo [s]'); plt.ylabel('Z [m]'); plt.grid(True); plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, "ultima_grafica_xyz_tello.png"))
        plt.close()

        # 3. Guardar PNG de Trayectoria 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([0], [0], [0], 'go', label='Inicio')
        ax.plot(self.tx_history, self.ty_history, self.tz_history, 'r--', label='Meta')
        ax.plot(self.x_history, self.y_history, self.z_history, 'b-', label='Trayecto 3D')
        ax.set_title('Trayectoria 3D - Tello')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        plt.savefig(os.path.join(dir_path, "ultima_trayectoria_3d_tello.png"))
        plt.close()
        
        self.get_logger().info(f"✅ Reportes de gráficas actualizados en: {dir_path}")

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TelloPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Deteniendo graficador y generando archivos finales...")
        node.exportar_datos()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
