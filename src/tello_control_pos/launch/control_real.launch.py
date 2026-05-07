import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction, RegisterEventHandler
from launch.event_handlers import OnShutdown

def generate_launch_description():
    pkg_dir = get_package_share_directory('tello_control_pos')
    ekf_config_path = os.path.join(pkg_dir, 'config', 'ekf.yaml')
    optitrack_config_path = os.path.join(pkg_dir, 'config', 'optitrack.yaml')

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
        
        # 2. Inyector de Covarianza (Escucha al dron real y añade ruido para el EKF)
        # Node(
        #     package='tello_control_pos',
        #     executable='odometry_integrator',
        #     name='covariance_injector',
        #     output='screen',
        #     remappings=[
        #         ('/drone1/odom', '/odom'),      # Remapeo al tópico del dron real
        #         ('/drone_pose', '/drone/pose')      # Remapeo al tópico del OptiTrack real
        #     ]
        # ),
        
        # 3. Ejecutar EKF (robot_localization)
        # Node(
        #     package='robot_localization',
        #     executable='ekf_node',
        #     name='ekf_filter_node',
        #     output='screen',
        #     parameters=[
        #         ekf_config_path,
        #         optitrack_config_path,
        #         {'use_sim_time': False}
        #     ]
        # ),
        
        # 4. Graficador
        Node(
            package='tello_control_pos',
            executable='plotter',
            name='plotter',
            output='screen'
        ),
        
        # 5. Controlador de Posición
        # Node(
        #     package='tello_control_pos',
        #     executable='position_controller',
        #     name='position_controller',
        #     output='screen',
        #     remappings=[
        #         ('/drone1/cmd_vel', '/control')  # Remapeo al tópico del dron real
        #     ],
        #     parameters=[
        #         {'velocity_scale': 100.0}
        #     ]
        # ),
        
        # 6. Enviar comando de takeoff al dron real (Retrasado para asegurar conexión)
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
        ),
        
        # 7. Al cerrar con Ctrl+C, enviar comando de aterrizaje
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[
                    ExecuteProcess(
                        cmd=[
                            'ros2', 'topic', 'pub', '--once',
                            '/land',
                            'std_msgs/msg/Empty',
                            '{}'
                        ],
                        output='screen'
                    )
                ]
            )
        )
    ])
