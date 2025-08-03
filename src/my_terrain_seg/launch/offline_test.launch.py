#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def launch_setup(context, *args, **kwargs):
    # Launch arguments
    use_rviz = LaunchConfiguration('use_rviz')
    offline_test = LaunchConfiguration('offline_test')
    
    nodes = []
    
    # PLY Publisher (for offline testing)
    nodes.append(Node(
        package='my_terrain_seg',
        executable='ply_publisher',
        name='ply_publisher',
        condition=IfCondition(offline_test),
        remappings=[
            ('/cloud_raw', '/cloud_in')  # Remap to segmentation input
        ],
        parameters=[{
            'ply_file': '/home/vincent/ros2_ws/data/ply/ep020_sample0.ply',
            'publish_rate': 0.5  # Publish every 2 seconds
        }]
    ))
    
    # Segmentation Node
    nodes.append(Node(
        package='my_terrain_seg',
        executable='segmentation_node',
        name='segmentation_node',
        parameters=[{
            'model_path': '/home/vincent/ros2_ws/src/my_terrain_seg/checkpoints/best.pth'
        }]
    ))
    
    # Octomap Filter Node
    nodes.append(Node(
        package='my_terrain_seg',
        executable='octomap_node',
        name='octomap_filter'
    ))
    
    # Octomap Server
    nodes.append(Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        parameters=[{
            'resolution': 0.05,
            'frame_id': 'map',
            'height_map': True,
            'color_free': True,
            'publish_free_space': True,
            'filter_ground': False,
            'base_frame_id': 'map',
            'filter_speckles': True,
            'occupancy_min_z': -1.0,
            'occupancy_max_z': 3.0
        }],
        remappings=[
            ('/cloud_in', '/occupied_points')
        ]
    ))
    
    # Static transform for map frame
    nodes.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_frame_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
    ))
    
    # RViz
    nodes.append(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        condition=IfCondition(use_rviz),
        arguments=['-d', '/home/vincent/ros2_ws/src/my_terrain_seg/config/stair_vis.rviz']
    ))
    
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz2 for visualization'
        ),
        DeclareLaunchArgument(
            'offline_test',
            default_value='false',
            description='Use PLY publisher for offline testing'
        ),
        OpaqueFunction(function=launch_setup)
    ])
