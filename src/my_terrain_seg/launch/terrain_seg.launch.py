# launch/terrain_seg.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # allow overriding model path and map resolution at launch time
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RViz'
    )
    
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/vincent/ros2_ws/src/my_terrain_seg/checkpoints/best.pth',
        description='Absolute path to your trained best.pth checkpoint'
    )
    res_arg = DeclareLaunchArgument(
        'resolution',
        default_value='0.05',
        description='OctoMap resolution (meters)'
    )
    frame_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='map',
        description='Frame ID for the OctoMap'
    )

    seg_node = Node(
        package='my_terrain_seg',
        executable='segmentation_node',
        name='terrain_segmentation',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path')
        }]
    )

    filter_node = Node(
        package='my_terrain_seg',
        executable='octomap_node',
        name='terrain_octomap',
        output='screen',
        parameters=[{
            'resolution': LaunchConfiguration('resolution')
        }]
    )

    octomap_server = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[{
            'resolution': LaunchConfiguration('resolution'),
            'frame_id': LaunchConfiguration('frame_id'),
            'height_map': True,
            'colored_map': False,  # Disable colors to avoid errors
            'filter_ground': False,  # we handle this in our segmentation
            'base_frame_id': LaunchConfiguration('frame_id'),
            'pointcloud_min_z': -1.0,  # Adjust for typical stair heights
            'pointcloud_max_z': 3.0,   # Adjust for typical stair heights
            'occupancy_min_z': -1.0,
            'occupancy_max_z': 3.0
        }],
        remappings=[
            ('cloud_in', '/cloud_occupied')
        ]
    )

    # Launch RViz with our configuration
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', str(os.path.join(
            get_package_share_directory('my_terrain_seg'),
            'config',
            'stair_vis.rviz'
        ))],
        additional_env={'QT_QPA_PLATFORM': 'xcb'}  # Force X11 backend
    )

    nodes = [
        model_arg,
        rviz_arg,
        res_arg,
        frame_arg,
        seg_node,
        filter_node,
        octomap_server,
    ]
    
    # Only add RViz node if requested
    if LaunchConfiguration('rviz'):
        nodes.append(rviz_node)
    
    return LaunchDescription(nodes)
