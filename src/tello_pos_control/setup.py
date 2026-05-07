from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tello_pos_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noe',
    maintainer_email='noe.benjamin2010@hotmail.com',
    description='Your package description here',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "pos_control = tello_pos_control.pos_controller_node:main",
            "odometry = tello_pos_control.odom_node:main",
            "reference = tello_pos_control.reference_node:main",
        ],
    },
)
