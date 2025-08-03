from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    ply_file_arg = DeclareLaunchArgument(
        'ply_file',
        default_value='/home/vincent/ros2_ws/data/ply/Stair_1.ply',
        description='Path to the PLY file to load'
    )
    
    loop_playback = DeclareLaunchArgument(
        'loop_playback',
        default_value='true',
        description='Loop the PLY file playback'
    )
    
    return LaunchDescription([
        ply_file_arg,
        loop_playback,

        # Static TF publisher for map -> base_link -> livox_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_base',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_livox',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'livox_frame']
        ),
        
        # PLY Publisher (simulates LiDAR data)
        Node(
            package='my_terrain_seg',
            executable='ply_publisher',
            name='ply_publisher',
            output='screen',
            parameters=[{
                'ply_file': LaunchConfiguration('ply_file'),
                'loop_playback': LaunchConfiguration('loop_playback'),
                'publish_rate': 1.0,
                'frame_id': 'map'
            }]
        ),

        # Segmentation Node (AI stair detection)
        Node(
            package='my_terrain_seg',
            executable='segmentation_node',
            name='segmentation_node',
            output='screen',
            remappings=[
                ('/cloud_raw', '/cloud_raw'),
                ('/cloud_seg', '/cloud_seg')
            ]
        ),

        # Real OctoMap Node (with actual octree implementation)
        Node(
            package='my_terrain_seg',
            executable='real_octomap_node',
            name='real_octomap_node',
            output='screen',
            parameters=[{
                'frame_id': 'map'  # Use map frame for consistency
            }],
            remappings=[
                ('/cloud_seg', '/cloud_seg'),
                ('/cloud_occupied', '/cloud_occupied')
            ]
        ),
    ])
