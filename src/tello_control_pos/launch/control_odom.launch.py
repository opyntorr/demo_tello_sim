import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    pkg_dir = get_package_share_directory('tello_control_pos')

    return LaunchDescription([
        # 1. Ejecutar plotter
        Node(
            package='tello_control_pos',
            executable='plotter',
            name='plotter',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
        
        # 2. Ejecutar position_controller usando solo odometría
        Node(
            package='tello_control_pos',
            executable='position_controller',
            name='position_controller',
            output='screen',
            remappings=[
                ('/odometry/filtered', '/drone1/odom')  # Remapear para que use odometría pura
            ],
            parameters=[
                {'use_sim_time': True},
                {'velocity_scale': 1.0},
                {'kp': 0.4},       # PID más suave para Gazebo
                {'kd': 0.15}       # PID más suave para Gazebo
            ]
        ),
        
        # 3. Enviar comando de takeoff con 3 segundos de retraso
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'service', 'call', 
                        '/drone1/tello_action', 
                        'tello_msgs/srv/TelloAction', 
                        "{cmd: 'takeoff'}"
                    ],
                    output='screen'
                )
            ]
        )
    ])
