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
        
        # PLY Publisher (simulates LiDAR data)
        Node(
            package='my_terrain_seg',
            executable='ply_publisher',
            name='ply_publisher',
            output='screen',
            parameters=[{
                'ply_file': LaunchConfiguration('ply_file'),
                'loop_playback': False,  # Changed to False to prevent continuous republishing
                'publish_rate': 0.1,     # Very slow rate since we only need it once
                'frame_id': 'livox_frame'
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
            remappings=[
                ('/cloud_seg', '/cloud_seg'),
                ('/cloud_occupied', '/cloud_occupied')
            ]
        ),

        # Add octomap_server Node to build full occupancy map with free space
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[
                {'frame_id': 'map'},              # Use map frame for consistency
                {'base_frame_id': 'base_link'},   # Base frame for TF
                {'resolution': 0.2},              # Good balance of detail vs performance
                {'sensor_model/max_range': 8.0},  # Reasonable range limit
                {'sensor_model/hit': 0.7},
                {'sensor_model/miss': 0.4},
                {'sensor_model/min': 0.12},
                {'sensor_model/max': 0.97},
                {'publish_free_space': True},     # Enable free space visualization
                {'latch': True},                  # Keep publishing even when not updating
                {'incremental_2D_projection': False}, # Disable incremental updates to reduce flashing
                {'filter_ground': False},         # Don't filter out ground points
                {'height_map': True},            # Generate 2D height map
                {'color_factor': 0.8},           # Enable colored visualization
                {'filter_speckles': True},       # Remove noise
                {'compress_map': False},         # Don't compress for better visualization
                {'point_subsample': 3},          # Moderate subsampling
                {'min_range': 0.3},              # Filter very close points
                {'occupancy_min_z': -1.5},      # Limit Z range
                {'occupancy_max_z': 3.0},
                {'use_sim_time': False}
            ],
            remappings=[
                ('/cloud_in', '/cloud_occupied'),  # Subscribe to your filtered stairs cloud
                ('/octomap_binary', '/octomap_binary'),
                ('/octomap_full', '/octomap_full'),
                ('/projected_map', '/projected_map')
            ]
        ),

        # Add TF publishers for proper coordinate frames
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
    ])
