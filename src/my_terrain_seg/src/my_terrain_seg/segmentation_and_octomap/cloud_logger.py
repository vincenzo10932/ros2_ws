#!/usr/bin/env python3
"""
ROS2 Point Cloud Logger - Save live data to PLY files
Usage: ros2 run my_terrain_seg cloud_logger --ros-args -p topic:=/cloud_seg -p filename:=stairs_room_A.ply
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os
from datetime import datetime

class CloudLogger(Node):
    def __init__(self):
        super().__init__('cloud_logger')
        
        # Parameters
        self.declare_parameter('topic', '/cloud_seg')
        self.declare_parameter('filename', 'captured_stairs.ply')
        self.declare_parameter('save_directory', '/home/vincent/ros2_ws/data/captured/')
        
        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.filename = self.get_parameter('filename').get_parameter_value().string_value
        self.save_dir = self.get_parameter('save_directory').get_parameter_value().string_value
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Storage for accumulated points
        self.accumulated_points = []
        self.accumulated_labels = []
        
        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2, topic, self.cloud_callback, 10)
        
        self.get_logger().info(f"Cloud logger started. Listening to {topic}")
        self.get_logger().info("Press Ctrl+C to save and exit")
        
    def cloud_callback(self, msg):
        """Accumulate point cloud data"""
        try:
            points = []
            labels = []
            
            # Extract points and labels from message
            for point in pc2.read_points(msg, field_names=["x", "y", "z", "label"], skip_nans=True):
                points.append([point[0], point[1], point[2]])
                labels.append(point[3])
            
            if len(points) > 0:
                self.accumulated_points.extend(points)
                self.accumulated_labels.extend(labels)
                
                self.get_logger().info(f"Accumulated {len(self.accumulated_points)} points so far...")
                
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")
    
    def save_ply(self):
        """Save accumulated points to PLY file"""
        if len(self.accumulated_points) == 0:
            self.get_logger().warn("No points to save!")
            return
            
        try:
            points = np.array(self.accumulated_points)
            labels = np.array(self.accumulated_labels)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = self.filename.replace('.ply', '')
            filename = f"{base_name}_{timestamp}.ply"
            filepath = os.path.join(self.save_dir, filename)
            
            # Write PLY file
            with open(filepath, 'w') as f:
                # PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property int label\n")
                f.write("end_header\n")
                
                # Point data
                for i, (point, label) in enumerate(zip(points, labels)):
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(label)}\n")
            
            self.get_logger().info(f"Saved {len(points)} points to {filepath}")
            self.get_logger().info(f"Labels: {np.bincount(labels.astype(int))}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving PLY file: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = CloudLogger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Saving captured data...")
        node.save_ply()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
