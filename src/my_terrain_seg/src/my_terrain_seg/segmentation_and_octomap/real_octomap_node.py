#!/usr/bin/env python3
"""
ROS2 Real OctoMap Node with Octree Implementation
This script:
 1. Subscribes to segmented point cloud from AI model
 2. Filters stair points (classes 1&2)
 3. Builds REAL OctoMap octr        self.last_octree_hash = None   # Hash of octree state to detect changes
        self.cloud_published = False   # Track if we've published the point cloud
        self.centers_published = False # Track if we've published voxel centers
        self.stable_timestamp = None   # Use stable timestamp to prevent flashing with occupied/free voxels using raytracing
 4. Publishes proper OctoMap binary data
 5. Shows real-time console-based visualization stats
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from octomap_msgs.msg import Octomap
from geometry_msgs.msg import Point
import numpy as np
import threading
import time
import struct
from collections import defaultdict

# Import hyperparameters
try:
    from my_terrain_seg.my_model_project.hyperparameters import OCTOMAP_CHUNK_SIZE, OCTOMAP_RESOLUTION
except ImportError:
    OCTOMAP_CHUNK_SIZE = 6000  # fallback
    OCTOMAP_RESOLUTION = 0.05  # fallback

class SimpleOctree:
    """
    Simplified Octree implementation for probabilistic occupancy mapping
    Implements raytracing, probabilistic updates, and binary serialization
    """
    def __init__(self, resolution=0.1, prob_hit=0.7, prob_miss=0.3, 
                 prob_thresh_min=0.12, prob_thresh_max=0.97):
        self.resolution = resolution
        self.prob_hit = prob_hit  
        self.prob_miss = prob_miss
        self.prob_thresh_min = prob_thresh_min
        self.prob_thresh_max = prob_thresh_max
        
        # Voxel grid: key=(x,y,z) -> probability
        self.voxels = defaultdict(lambda: 0.5)  # Start with unknown (0.5)
        self.sensor_origin = np.array([0.0, 0.0, 0.0])
        
    def world_to_key(self, point):
        """Convert world coordinates to voxel key"""
        return tuple(np.floor(np.array(point, dtype=np.float64) / self.resolution).astype(np.int32))
        
    def key_to_world(self, key):
        """Convert voxel key to world coordinates (center)"""
        return np.array(key, dtype=np.float64) * self.resolution + self.resolution/2
        
    def insert_point_cloud(self, points, sensor_origin=None):
        """Insert point cloud with simplified approach for speed"""
        if sensor_origin is None:
            sensor_origin = self.sensor_origin
        else:
            self.sensor_origin = np.array(sensor_origin)
        
        # Fast mode: Just mark occupied voxels without raytracing
        # This is much faster for terrain segmentation where we trust the AI classification
        for point in points:
            endpoint_key = self.world_to_key(point)
            self.update_voxel(endpoint_key, is_occupied=True)
    
    def insert_point_cloud_with_raytracing(self, points, sensor_origin=None):
        """Insert point cloud with full raytracing for free space (slower but more accurate)"""
        if sensor_origin is None:
            sensor_origin = self.sensor_origin
        else:
            self.sensor_origin = np.array(sensor_origin)
            
        for point in points:
            self.insert_ray(sensor_origin, point)
    
    def insert_ray(self, origin, endpoint):
        """Insert ray from origin to endpoint, updating free and occupied voxels"""
        # Discretize ray using Bresenham-like 3D algorithm
        ray_voxels = self.raytrace_3d(origin, endpoint)
        
        # Update free space (all voxels along ray except endpoint)
        for voxel_key in ray_voxels[:-1]:
            self.update_voxel(voxel_key, is_occupied=False)
            
        # Update occupied space (endpoint)
        endpoint_key = self.world_to_key(endpoint)
        self.update_voxel(endpoint_key, is_occupied=True)
    
    def raytrace_3d(self, start, end):
        """3D raytracing to get all voxels along the ray"""
        start_key = self.world_to_key(start)
        end_key = self.world_to_key(end)
        
        # Simple 3D DDA algorithm
        voxels = []
        current = np.array(start_key, dtype=np.float64)
        target = np.array(end_key, dtype=np.float64)
        
        # Direction and step size
        direction = target - current
        if np.allclose(direction, 0):
            return [tuple(current.astype(np.int32))]
            
        steps = max(abs(direction))
        if steps == 0:
            return [tuple(current.astype(np.int32))]
            
        step_size = direction / steps
        
        for i in range(int(steps) + 1):
            voxel_key = tuple(np.round(current).astype(np.int32))
            voxels.append(voxel_key)
            current = current + step_size
            
        return voxels
    
    def update_voxel(self, voxel_key, is_occupied):
        """Update voxel probability using Bayesian update"""
        current_prob = self.voxels[voxel_key]
        
        if is_occupied:
            # Hit: increase probability
            new_prob = current_prob + self.prob_hit * (1 - current_prob)
        else:
            # Miss: decrease probability  
            new_prob = current_prob * (1 - self.prob_miss)
            
        # Clamp to thresholds
        self.voxels[voxel_key] = np.clip(new_prob, self.prob_thresh_min, self.prob_thresh_max)
    
    def get_occupied_voxels(self, threshold=0.5):
        """Get list of occupied voxel centers"""
        occupied = []
        for key, prob in self.voxels.items():
            if prob > threshold:
                occupied.append(self.key_to_world(key))
        return np.array(occupied) if occupied else np.empty((0, 3))
    
    def get_free_voxels(self, threshold=0.5):
        """Get list of free voxel centers"""
        free = []
        for key, prob in self.voxels.items():
            if prob < threshold:
                free.append(self.key_to_world(key))
        return np.array(free) if free else np.empty((0, 3))
    
    def to_binary_data(self):
        """Convert octree to binary format for OctoMap message"""
        # Simplified binary format: resolution + voxel count + voxel data
        data = struct.pack('f', self.resolution)  # 4 bytes: resolution
        data += struct.pack('I', len(self.voxels))  # 4 bytes: voxel count
        
        for key, prob in self.voxels.items():
            # Pack: x, y, z (3 ints), probability (1 float)
            data += struct.pack('iiif', key[0], key[1], key[2], prob)
        
        # Convert to list of unsigned bytes, then convert to signed int8 range
        byte_list = []
        for byte_val in data:
            # Convert unsigned byte (0-255) to signed int8 (-128 to 127)
            if byte_val > 127:
                byte_list.append(byte_val - 256)
            else:
                byte_list.append(byte_val)
        
        return byte_list
    
    def get_bounds(self):
        """Get bounding box of the octree"""
        if not self.voxels:
            return np.zeros(6)  # [x_min, x_max, y_min, y_max, z_min, z_max]
            
        keys = np.array(list(self.voxels.keys()))
        world_coords = keys * self.resolution
        
        return np.array([
            world_coords[:, 0].min(), world_coords[:, 0].max(),
            world_coords[:, 1].min(), world_coords[:, 1].max(), 
            world_coords[:, 2].min(), world_coords[:, 2].max()
        ])

def pointcloud_from_ros(cloud_msg):
    """
    Convert ROS PointCloud2 message to numpy array with labels.
    Returns: points (N,3), labels (N,) where labels are segmentation classes
    """
    points = []
    labels = []
    
    for point in pc2.read_points(cloud_msg, field_names=['x', 'y', 'z', 'label'], skip_nans=True):
        points.append([float(point[0]), float(point[1]), float(point[2])])
        labels.append(int(point[3]))
    
    return np.array(points, dtype=np.float64), np.array(labels, dtype=np.int32)

class RealOctomapNode(Node):
    def __init__(self):
        super().__init__('real_octomap_node')
        
        # Initialize real octree - FILTERING FOR STAIRS ONLY
        self.octree = SimpleOctree(
            resolution=0.15,         # 15cm voxels - good for stair steps (~30cm typical)
            prob_hit=0.6,           # Lower hit probability - requires more evidence
            prob_miss=0.4,          # Higher miss probability - more wall removal
            prob_thresh_min=0.12,   # Standard minimum threshold
            prob_thresh_max=0.85    # Lower max threshold - more selective
        )
        
        # ROS Publishers and Subscribers
        self.cloud_sub = self.create_subscription(
            PointCloud2, '/cloud_seg', self.seg_callback, 10)
        self.cloud_pub = self.create_publisher(
            PointCloud2, '/cloud_occupied', 10)
        self.octomap_pub = self.create_publisher(
            Octomap, '/octomap_real', 10)
        
        # Sensor parameters for raytracing
        self.sensor_origin = np.array([0.0, 0.0, 2.0])  # Elevated sensor position for more coverage
        self.frame_id = 'map'  # Use map frame for consistency with octomap_server
        self.occupied_points = []  # Store occupied points for visualization
        
        # Track changes to avoid unnecessary republishing
        self.last_voxel_count = 0
        self.last_publish_time = 0
        self.octree_published = False  # Track if published the final octree
        self.last_octree_hash = None   # Hash of octree state to detect changes
        self.stable_timestamp = None   # Use stable timestamp to prevent flashing
        self.cloud_publish_counter = 0  # Track cloud publications
        
        # Statistics
        self.total_points_processed = 0
        self.total_rays_traced = 0
        
        self.get_logger().info("Real OctoMap Node with Octree initialized")

    def seg_callback(self, msg):
        """Process segmented point cloud from AI model"""
        try:
            self.get_logger().info("=== Starting seg_callback ===")
            
            # Convert ROS message to numpy arrays
            self.get_logger().info("Converting ROS message to numpy arrays")
            points, labels = pointcloud_from_ros(msg)
            
            if len(points) == 0:
                self.get_logger().warn("No points in message, returning")
                return
                
            self.get_logger().info(f"Converted {len(points)} points with {len(labels)} labels")
            
            # Filter for stair points only (classes 1=risers, 2=treads)
            stair_mask = (labels == 1) | (labels == 2)
            stair_points = points[stair_mask]
            stair_labels = labels[stair_mask]
            
            self.get_logger().info(f"After stair filtering: {len(stair_points)} stair points")
            
            # Debug: Log label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_info = ", ".join([f"label_{label}: {count}" for label, count in zip(unique_labels, counts)])
            self.get_logger().info(f"Label distribution: {label_info}")
            
            # Debug: Check what we're filtering
            non_stair_points = points[~stair_mask]
            self.get_logger().info(f"Filtering OUT {len(non_stair_points)} non-stair points")
            self.get_logger().info(f"Keeping {len(stair_points)} stair points")
            
            # Filter out invalid coordinates (very large values, NaN, inf)
            if len(stair_points) > 0:
                self.get_logger().info("Filtering invalid coordinates")
                valid_mask = (
                    (np.abs(stair_points[:, 0]) < 100) &  # X within ±100m
                    (np.abs(stair_points[:, 1]) < 100) &  # Y within ±100m  
                    (np.abs(stair_points[:, 2]) < 100) &  # Z within ±100m
                    np.isfinite(stair_points).all(axis=1)  # No NaN or inf
                )
                stair_points = stair_points[valid_mask]
                stair_labels = stair_labels[valid_mask]
                self.get_logger().info(f"After coordinate filtering: {len(stair_points)} valid points")
            
            num_risers = np.sum(labels == 1)
            num_treads = np.sum(labels == 2)
            
            self.get_logger().info(
                f"Received {len(points)} segmented points")
            self.get_logger().info(
                f"Found {num_risers} risers, {num_treads} treads, {len(stair_points)} total occupied points")
            
            if len(stair_points) > 0:
                self.get_logger().info(f"Inserting {len(stair_points)} points into octree (chunked processing)")
                
                # Process octree insertion in chunks for better performance
                chunk_size = OCTOMAP_CHUNK_SIZE
                total_chunks = (len(stair_points) + chunk_size - 1) // chunk_size
                
                for i in range(0, len(stair_points), chunk_size):
                    chunk_end = min(i + chunk_size, len(stair_points))
                    chunk_points = stair_points[i:chunk_end]
                    chunk_num = (i // chunk_size) + 1
                    
                    self.get_logger().info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_points)} points)")
                    # Use raytracing for better wall filtering - slower but much better geometric filtering
                    self.octree.insert_point_cloud_with_raytracing(chunk_points, self.sensor_origin)
                    
                self.total_points_processed += len(stair_points)
                self.total_rays_traced += len(stair_points)
                
                self.get_logger().info("Updating octree")
                # Update octree and publish (less frequent to reduce MarkerArray flashing)
                self.update_octree(stair_points)
                
                # Publish filtered cloud only when octree is stable
                self.get_logger().info("Publishing filtered cloud")
                # Publish geometrically cleaned voxel centers
                self.publish_filtered_cloud(stair_points, stair_labels, msg.header)
                
                self.get_logger().info("=== seg_callback completed successfully ===")
            else:
                self.get_logger().warn("No valid stair points to process")
                
        except Exception as e:
            self.get_logger().error(f"Error in seg_callback: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def get_octree_hash(self):
        """Generate a hash of the current octree state to detect changes"""
        import hashlib
        # Create a string representation of the octree state
        voxel_data = []
        for key, prob in sorted(self.octree.voxels.items()):
            if prob > 0.5:  # Only include occupied voxels
                voxel_data.append(f"{key[0]},{key[1]},{key[2]},{prob:.3f}")
        
        octree_string = "|".join(voxel_data)
        return hashlib.md5(octree_string.encode()).hexdigest()

    def update_octree(self, points):
        """Create and publish real OctoMap message"""
        try:
            # Store points for visualization
            self.occupied_points = points
            
            # Generate hash of current octree state
            current_hash = self.get_octree_hash()
            
            # Check if octree has actually changed
            octree_changed = (current_hash != self.last_octree_hash)
            current_time = time.time()
            time_since_last = current_time - self.last_publish_time
            
            # Only publish if significant changes or first time (reduce MarkerArray flashing)
            should_publish = (
                (octree_changed and time_since_last > 2.0) or  # Wait 2s between updates
                (time_since_last > 15.0 and not self.octree_published) or  # Max 15s interval for initial build
                self.last_publish_time == 0  # First time publishing
            )
            
            if should_publish:
                # Create real OctoMap message with binary data
                octomap_msg = Octomap()
                octomap_msg.header.frame_id = self.frame_id
                
                # Use stable timestamp to prevent MarkerArray flashing
                if self.stable_timestamp is None:
                    self.stable_timestamp = self.get_clock().now().to_msg()
                octomap_msg.header.stamp = self.stable_timestamp
                
                octomap_msg.resolution = self.octree.resolution
                
                # Convert octree to binary data
                octomap_msg.data = self.octree.to_binary_data()
                octomap_msg.binary = True
                
                self.octomap_pub.publish(octomap_msg)
                
                # Update tracking variables
                self.last_octree_hash = current_hash
                self.last_publish_time = current_time
                self.octree_published = True
                
                # Log statistics
                occupied_voxels = self.octree.get_occupied_voxels()
                free_voxels = self.octree.get_free_voxels()
                
                self.get_logger().info(
                    f"Published octree update: {len(self.octree.voxels)} total voxels, "
                    f"{len(occupied_voxels)} occupied, {len(free_voxels)} free")
            else:
                self.get_logger().debug(
                    f"Octree unchanged - skipping publish (time: {time_since_last:.1f}s)")
            
        except Exception as e:
            self.get_logger().error(f"Error updating octree: {e}")
            
    def publish_filtered_cloud(self, points, labels, header):
        """Publish geometrically cleaned octree voxel centers as PointCloud2"""
        try:
            # Get occupied voxel centers from octree (geometrically cleaned data)
            # Higher threshold = more aggressive filtering = fewer points = stairs only
            occupied_voxels = self.octree.get_occupied_voxels(threshold=0.75)
            
            if len(occupied_voxels) == 0:
                self.get_logger().warn("No occupied voxels to publish to /cloud_occupied")
                return
            
            # Create PointCloud2 message
            cloud_msg = PointCloud2()
            cloud_msg.header = header
            cloud_msg.header.frame_id = 'map'  # Use map frame for octomap_server consistency
            
            # Use stable timestamp to prevent flashing in RViz
            if self.stable_timestamp is None:
                self.stable_timestamp = self.get_clock().now().to_msg()
            cloud_msg.header.stamp = self.stable_timestamp
            
            # Create point data from voxel centers (no labels needed - just geometry)
            output_points = []
            for voxel_center in occupied_voxels:
                if np.all(np.isfinite(voxel_center)):
                    # Use label=3 to distinguish from raw AI classes (1=risers, 2=treads)
                    output_points.append([voxel_center[0], voxel_center[1], voxel_center[2], 3.0])
            
            if output_points:
                # Create fields including label for color mapping
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1), 
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='label', offset=12, datatype=PointField.FLOAT32, count=1)
                ]
                
                cloud_msg = pc2.create_cloud(cloud_msg.header, fields, output_points)
                self.cloud_pub.publish(cloud_msg)
                
                # Increment counter and log regularly
                self.cloud_publish_counter += 1
                self.get_logger().info(f"✅ Published /cloud_occupied #{self.cloud_publish_counter} with {len(output_points)} voxel centers (cleaned geometry)")
            else:
                self.get_logger().warn("No valid voxel centers to publish to /cloud_occupied")
            
        except Exception as e:
            self.get_logger().error(f"❌ Error publishing /cloud_occupied: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

def main(args=None):
    rclpy.init(args=args)
    node = RealOctomapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
