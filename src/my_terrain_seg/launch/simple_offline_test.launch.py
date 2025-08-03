#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz2 for visualization'
        ),
        
        # PLY Publisher - loads your sample stair data
        Node(
            package='my_terrain_seg',
            executable='ply_publisher',
            name='ply_publisher',
            remappings=[
                ('/cloud_raw', '/cloud_in')  # Feed directly to segmentation
            ],
            parameters=[{
                'ply_file': '/home/vincent/ros2_ws/data/ply/ep020_sample0.ply',
                'publish_rate': 0.5  # Publish every 2 seconds
            }]
        ),
        
        # Segmentation Node - your AI model analyzing the stairs
        Node(
            package='my_terrain_seg',
            executable='segmentation_node',
            name='segmentation_node',
            parameters=[{
                'model_path': '/home/vincent/ros2_ws/src/my_terrain_seg/checkpoints/best.pth'
            }]
        ),
        
        # Static transform for coordinate frames
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_frame_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
        ),
        
        # RViz for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            arguments=['-d', '/home/vincent/ros2_ws/src/my_terrain_seg/config/stair_vis.rviz']
        )
    ])
