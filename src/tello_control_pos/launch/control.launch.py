import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    pkg_dir = get_package_share_directory('tello_control_pos')
    ekf_config_path = os.path.join(pkg_dir, 'config', 'ekf.yaml')

    return LaunchDescription([
        # 1. Ejecutar inyector de covarianza
        Node(
            package='tello_control_pos',
            executable='odometry_integrator',
            name='covariance_injector',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'vel_multiplier': 1.0}
            ]
        ),
        
        # 2. Ejecutar EKF (robot_localization)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                ekf_config_path,
                {'use_sim_time': True}
            ]
        ),
        
        # 3. Ejecutar plotter
        Node(
            package='tello_control_pos',
            executable='plotter',
            name='plotter',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
        
        # 4. Ejecutar position_controller
        Node(
            package='tello_control_pos',
            executable='position_controller',
            name='position_controller',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'velocity_scale': 1.0},
                {'kp': 0.4},       # PID más suave para Gazebo
                {'kd': 0.15}       # PID más suave para Gazebo
            ]
        ),
        
        # 5. Enviar comando de takeoff con 3 segundos de retraso para asegurar que la simulación esté lista
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
