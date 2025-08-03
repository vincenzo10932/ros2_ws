#!/usr/bin/env python3
"""
Real LiDAR Hardware Deployment - Production System
Livox Mid-360 LiDAR + AI Stair Detection + RTABMap SLAM + Octomap
For live autonomous navigation in stair environments
"""

from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
        
        # ============ SENSOR INPUT ============
        # Livox ROS2 Driver (connects to hardware)
        Node(
            package='livox_ros_driver2',
            executable='livox_ros_driver2_node',
            name='livox_driver',
            output='screen',
            parameters=[{
                'frame_id': 'livox_frame',
                'lidar_bag': False,
                'xfer_format': 1,
                'multi_topic': 0,
                'data_src': 0,
                'publish_freq': 10.0,
                'output_data_type': 0,
                'cmdline_bd_code': 'livox0000000001'
            }],
            remappings=[
                ('/livox/lidar', '/livox/lidar_points')
            ]
        ),

        # ============ AI STAIR DETECTION ============
        # Segmentation Node (AI stair detection)
        Node(
            package='my_terrain_seg',
            executable='segmentation_node',
            name='segmentation_node',
            output='screen',
            remappings=[
                ('/cloud_raw', '/livox/lidar_points'),  # Input from hardware LiDAR
                ('/cloud_seg', '/cloud_seg')            # Output segmented stairs
            ]
        ),

        # ============ COORDINATE FRAMES ============
        # TF: map -> base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_base_link',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
        ),
        
        # TF: base_link -> livox_frame  
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_livox',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'livox_frame']
        ),

        # ============ OCTOMAP 3D MAPPING ============
        # OctoMap Server (3D occupancy mapping with full visualization)
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[{
                'resolution': 0.2,
                'frame_id': 'map',
                'sensor_model.max_range': 25.0,
                'sensor_model.min_range': 0.5,
                'sensor_model.hit_prob': 0.9,
                'sensor_model.miss_prob': 0.1,
                'sensor_model.clamping_thres_min': 0.12,
                'sensor_model.clamping_thres_max': 0.97,
                'publish_free_space': True,        # Enable free space visualization
                'latch': True,
                'height_map': True,               # Enable 2D height map projection
                'colored_map': False,
                'occupancy_min_z': -2.0,
                'occupancy_max_z': 3.0,
                'filter_speckles': True,
                'filter_ground': False,
                'ground_filter.distance': 0.04,
                'ground_filter.angle': 0.15,
                'ground_filter.plane_distance': 0.07,
                'compress_map': True,
                'incremental_2D_projection': True
            }],
            remappings=[
                ('cloud_in', '/cloud_seg')
            ]
        ),

        # ============ RTABMAP SLAM (GLOBAL MAPPING) ============
        # RTABMap SLAM node for global localization and mapping
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap_slam',
            output='screen',
            parameters=[
                {
                    'frame_id': 'livox_frame',
                    'subscribe_depth': False,
                    'subscribe_rgb': False,
                    'subscribe_scan_cloud': True,
                    'approx_sync': False,
                    
                    # RTABMap parameters optimized for LiDAR SLAM
                    'Rtabmap/DetectionRate': '1.0',
                    'Rtabmap/CreateIntermediateNodes': 'true',
                    'RGBD/NeighborLinkRefining': 'false',
                    'RGBD/ProximityBySpace': 'true',
                    'RGBD/ProximityMaxGraphDepth': '0',
                    'RGBD/ProximityPathMaxNeighbors': '1',
                    'RGBD/LocalRadius': '10.0',
                    'RGBD/AngularUpdate': '0.1',
                    'RGBD/LinearUpdate': '0.1',
                    'Mem/NotLinkedNodesKept': 'false',
                    'Mem/STMSize': '30',
                    'Mem/LaserScanDownsampleStepSize': '2',
                    
                    # ICP parameters for point cloud registration
                    'Icp/PM': 'true',
                    'Icp/PMOutlierRatio': '0.1',
                    'Icp/CorrespondenceRatio': '0.4',
                    'Icp/PointToPlane': 'true',
                    'Icp/PointToPlaneGroundNormalsUp': '0.8',
                    'Icp/MaxTranslation': '3.0',
                    'Icp/MaxRotation': '1.57',
                    'Icp/VoxelSize': '0.1',
                    'Icp/DownsamplingStep': '2',
                    'Icp/RangeMin': '0.5',
                    'Icp/RangeMax': '30.0',
                    
                    # Grid parameters for occupancy mapping
                    'Grid/FromDepth': 'false',
                    'Grid/3D': 'true',
                    'Grid/GroundIsObstacle': 'false',
                    'Grid/MaxObstacleHeight': '3.0',
                    'Grid/MaxGroundHeight': '0.2',
                    'Grid/NormalsSegmentation': 'false',
                    'Grid/CellSize': '0.05',
                    'Grid/RangeMin': '0.5',
                    'Grid/RangeMax': '30.0'
                }
            ],
            remappings=[
                ('cloud', '/livox/lidar_points'),
                ('scan_cloud', '/livox/lidar_points')
            ]
        ),

        # RTABMap Visualization (optional - disable for performance)
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            output='screen',
            parameters=[
                {
                    'frame_id': 'livox_frame',
                    'subscribe_scan_cloud': True,
                    'approx_sync': False
                }
            ],
            remappings=[
                ('cloud', '/livox/lidar_points')
            ]
        )
    ])
