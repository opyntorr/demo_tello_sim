import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    return LaunchDescription([
        # 1. Driver del Tello Real (Se conecta por WiFi al dron físico)
        Node(
            package='tello',
            executable='tello',
            name='tello_driver',
            output='screen',
            parameters=[
                {'tello_ip': '192.168.10.1'}  # IP por defecto del Tello
            ]
        ),
        
        # 2. Odometry Integrator (Escucha al dron real y calcula la posición)
        Node(
            package='tello_control_pos',
            executable='odometry_integrator',
            name='odometry_integrator',
            output='screen',
            remappings=[
                ('/drone1/odom', '/odom'),
                ('/drone1/imu', '/imu')
            ]
        ),
        
        # 3. Graficador
        Node(
            package='tello_control_pos',
            executable='plotter',
            name='plotter',
            output='screen'
        ),
        
        # 4. Controlador de Posición
        Node(
            package='tello_control_pos',
            executable='position_controller',
            name='position_controller',
            output='screen',
            remappings=[
                ('/drone1/cmd_vel', '/control')  # Remapeo al tópico del dron real
            ],
            parameters=[
                {'velocity_scale': 100.0}
            ]
        ),
        
        # 5. Enviar comando de takeoff al dron real (Retrasado para asegurar conexión)
        TimerAction(
            period=4.0,  # 4 segundos para darle tiempo al nodo tello de conectarse al WiFi
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'topic', 'pub', '--once', 
                        '/takeoff', 
                        'std_msgs/msg/Empty', 
                        '{}'
                    ],
                    output='screen'
                )
            ]
        )
    ])
