#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import EnvironmentVariable
import os

def generate_launch_description():
    # Ensure RMW implementation is set
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    
    # Create a group with a namespace to avoid conflicts
    return LaunchDescription([
        GroupAction(
            actions=[
                PushRosNamespace('lidar_system'),
                
                # TF publishers
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name='map_to_base_tf',
                    arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
                ),
                
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name='base_to_livox_tf',
                    arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'livox_frame']
                ),
                
                # PLY publisher
                Node(
                    package='my_terrain_seg',
                    executable='ply_publisher',
                    name='ply_publisher',
                    parameters=[{
                        'use_sim_time': False
                    }],
                    remappings=[
                        ('/cloud_raw', '/lidar_system/cloud_raw')
                    ]
                ),
                
                # Segmentation node
                Node(
                    package='my_terrain_seg',
                    executable='segmentation_node',
                    name='segmentation_node',
                    parameters=[{
                        'use_sim_time': False
                    }],
                    remappings=[
                        ('/cloud_in', '/lidar_system/cloud_raw'),
                        ('/cloud_seg', '/lidar_system/cloud_seg')
                    ]
                )
            ]
        )
    ])
