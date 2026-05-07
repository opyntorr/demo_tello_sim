from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    tello_odometry = Node(
        package="tello_pos_control",
        executable="odometry",
        output="screen",
    )
    tello_reference = Node(
        package="tello_pos_control",
        executable="reference",
        output="screen",
    )
    tello_control = Node(
        package="tello_pos_control",
        executable="pos_control",
        output="screen",
    )

    ld = LaunchDescription()
    ld.add_action(tello_odometry)
    ld.add_action(tello_reference)
    ld.add_action(tello_control)

    return ld
