#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Performance-optimized launch file for laptops with graphics limitations
    Reduces update rates and uses single-shot publishing
    """
    
    # Launch arguments
    ply_file_arg = DeclareLaunchArgument(
        'ply_file',
        default_value='/home/vincent/ros2_ws/data/ply/Stair_1.ply',
        description='Path to PLY file'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate', 
        default_value='0.05',  # Very slow for laptop performance
        description='Publishing rate in Hz'
    )
    
    loop_playback_arg = DeclareLaunchArgument(
        'loop_playback',
        default_value='false',  # Single-shot to prevent flashing
        description='Whether to loop playback'
    )

    # Nodes
    ply_publisher = Node(
        package='my_terrain_seg',
        executable='ply_publisher',
        name='ply_publisher',
        parameters=[{
            'ply_file': LaunchConfiguration('ply_file'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'loop_playback': LaunchConfiguration('loop_playback')
        }],
        output='screen'
    )

    segmentation_node = Node(
        package='my_terrain_seg',
        executable='segmentation_node',
        name='segmentation_node',
        parameters=[{
            'model_path': '/home/vincent/ros2_ws/src/my_terrain_seg/checkpoints/best.pth'
        }],
        output='screen'
    )

    real_octomap_node = Node(
        package='my_terrain_seg',
        executable='real_octomap_node', 
        name='real_octomap_node',
        output='screen'
    )

    return LaunchDescription([
        ply_file_arg,
        publish_rate_arg,
        loop_playback_arg,
        ply_publisher,
        segmentation_node,
        real_octomap_node,
    ])
