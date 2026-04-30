from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    return LaunchDescription([
        # 1. Ejecutar odometry_integrator
        Node(
            package='tello_control_pos',
            executable='odometry_integrator',
            name='odometry_integrator',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'vel_multiplier': 1.0}
            ]
        ),
        
        # 2. Ejecutar plotter
        Node(
            package='tello_control_pos',
            executable='plotter',
            name='plotter',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
        
        # 3. Ejecutar position_controller
        Node(
            package='tello_control_pos',
            executable='position_controller',
            name='position_controller',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'velocity_scale': 1.0}
            ]
        ),
        
        # 4. Enviar comando de takeoff con 3 segundos de retraso para asegurar que la simulación esté lista
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
