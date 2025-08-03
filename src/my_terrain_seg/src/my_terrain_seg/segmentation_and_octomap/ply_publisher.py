#!/usr/bin/env python3
"""
PLY Point Cloud Publisher

This node loads 3D point cloud data from .ply files and streams them to ROS2 topics.
It's the data source for the stair detection pipeline, providing realistic test data.

What it does:
1. Reads .ply files containing X,Y,Z coordinates and normal vectors (nx,ny,nz)
2. Converts the data to ROS2 PointCloud2 messages
3. Publishes point clouds on /cloud_raw topic at regular intervals
4. Supports single-shot or loop playback modes for testing

Key Feature: Includes normal vectors essential for geometric filtering
"""

# ─── Standard Library Imports ────────────────────────────────────────────────────
import os

# ─── Third-Party Imports ─────────────────────────────────────────────────────────
import numpy as np

# ─── ROS2 Imports ────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

class PLYPublisher(Node):
    """
    ROS2 node that loads and publishes 3D point cloud data from PLY files.
    
    Serves as the data source for testing the stair detection pipeline with
    realistic indoor scan data containing stairs, walls, and floors.
    """
    
    def __init__(self):
        super().__init__('ply_publisher')
        self.get_logger().info('📂 Initializing Point Cloud Data Publisher...')
        
        # ─── Configuration Parameters ────────────────────────────────────────────
        self.declare_parameter('ply_file', '/home/vincent/ros2_ws/data/ply/Stair_1.ply')
        self.declare_parameter('publish_rate', 0.1)  # Hz - Optimized for stable visualization
        self.declare_parameter('loop_playback', False)  # Single-shot mode prevents RViz flashing
        
        ply_file = self.get_parameter('ply_file').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.loop_playback = self.get_parameter('loop_playback').get_parameter_value().bool_value
        
        self.get_logger().info(f'📁 Loading point cloud: {ply_file}')
        self.get_logger().info(f'⏱️  Publishing rate: {publish_rate} Hz')
        self.get_logger().info(f'🔄 Loop mode: {self.loop_playback}')
        
        # ─── ROS2 Publishers Setup ───────────────────────────────────────────────
        # Publish raw point clouds for the segmentation pipeline
        self.pub_raw = self.create_publisher(PointCloud2, '/cloud_raw', 10)
        
        # ─── Point Cloud Data Loading ────────────────────────────────────────────
        if not os.path.exists(ply_file):
            self.get_logger().error(f'❌ PLY file not found: {ply_file}')
            self.get_logger().error('💡 Please check the file path and ensure the data exists')
            self.raw_points = None
            self.stair_points = None
            self.file_loaded = False
        else:
            # Initialize point cloud storage containers
            self.raw_points = None
            self.stair_points = None
            
            # Attempt to load the PLY file with error handling
            result = self.load_ply(ply_file)
            if result is None:
                self.get_logger().error('❌ Failed to load PLY file - node will continue running')
                self.raw_points = None
                self.stair_points = None
                self.file_loaded = False
            else:
                # Extract points from the result tuple (assuming load_ply returns raw_points, stair_points)
                self.raw_points, self.stair_points = result
                point_count = len(self.raw_points) if self.raw_points is not None else 0
                self.get_logger().info(f'✅ Successfully loaded {point_count} points from PLY file')
                self.file_loaded = True
        
        # ─── Publishing Timer Setup ──────────────────────────────────────────────
        # Always create timer to keep the node active (even if file loading failed)
        self.timer = self.create_timer(1.0/publish_rate, self.publish_clouds)
        self.publish_count = 0  # Track publications for single-shot mode
        
        # ─── Initialization Summary ──────────────────────────────────────────────
        if hasattr(self, 'file_loaded') and self.file_loaded:
            point_count = len(self.raw_points) if self.raw_points is not None else 0
            self.get_logger().info(f'🎯 PLY Publisher ready - loaded {point_count} points for processing')
        else:
            self.get_logger().warn('PLY Publisher initialized but no file loaded')
        
        self.get_logger().info(f'Publishing at {publish_rate} Hz, loop_playback: {self.loop_playback}')
        
    def load_ply(self, filename):
        """PLY loader supporting both ASCII and binary formats"""
        try:
            self.get_logger().info(f'Loading PLY file: {filename}')
            
            # Check file size for large files
            file_size = os.path.getsize(filename)
            self.get_logger().info(f'PLY file size: {file_size / (1024*1024):.1f} MB')
            
            # First, read the header to determine format
            with open(filename, 'rb') as f:
                header_lines = []
                while True:
                    line = f.readline().decode('utf-8', errors='ignore').strip()
                    header_lines.append(line)
                    if line == 'end_header':
                        break
                        
            header_text = '\n'.join(header_lines)
            
            # Check format
            is_binary = 'format binary' in header_text
            is_ascii = 'format ascii' in header_text
            
            if is_binary:
                self.get_logger().info('Detected binary PLY format')
                return self.load_binary_ply(filename, header_text)
            elif is_ascii:
                self.get_logger().info('Detected ASCII PLY format')
                return self.load_ascii_ply(filename, header_text)
            else:
                self.get_logger().error('Unknown PLY format')
                return None
                
        except Exception as e:
            self.get_logger().error(f'Error loading PLY: {str(e)}')
            return None
            
    def load_ascii_ply(self, filename, header_text):
        """Load ASCII format PLY with RGB colors"""
        try:
            with open(filename, 'r') as f:
                content = f.read()
                
            # Split header and data
            header_part, data_part = content.split('end_header', 1)
            
            # Parse vertex count
            vertex_count = self.parse_vertex_count(header_part)
            if vertex_count == 0:
                return None
                
            self.get_logger().info(f'Found {vertex_count} vertices in ASCII PLY')
                
            # Parse data points with RGB colors
            data_lines = data_part.strip().split('\n')
            raw_points = []
            stair_points = []
            
            for i, line in enumerate(data_lines):
                if i >= vertex_count:
                    break
                parts = line.strip().split()
                
                # Handle different PLY formats
                if len(parts) >= 9:  # x, y, z, r, g, b, nx, ny, nz
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                    nx, ny, nz = float(parts[6]), float(parts[7]), float(parts[8])
                    
                elif len(parts) >= 6:  # x, y, z, r, g, b (no normals)
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                    nx, ny, nz = 0.0, 0.0, 0.0  # Default normals
                    
                elif len(parts) >= 6:  # x, y, z, nx, ny, nz (no colors)  
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    nx, ny, nz = float(parts[3]), float(parts[4]), float(parts[5])
                    r, g, b = 128, 128, 128  # Default colors
                    
                elif len(parts) >= 3:  # x, y, z only
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = 128, 128, 128  # Default colors
                    nx, ny, nz = 0.0, 0.0, 0.0  # Default normals
                else:
                    continue
                    
                # Add to raw points with normal vectors
                raw_points.append([x, y, z, 128, nx, ny, nz])
                
                # Check if it's a stair point
                is_riser = r > 200 and g < 100  # Red - risers
                is_tread = g > 200 and r < 100  # Green - treads
                
                if is_riser or is_tread:  # Only include stairs
                    intensity = 255 if is_riser else 200  # Make stairs very visible
                    stair_points.append([x, y, z, intensity, nx, ny, nz])
                else:
                    # For files without color coding, include all points
                    stair_points.append([x, y, z, 200, nx, ny, nz])
                    
            self.get_logger().info(f'Parsed {len(raw_points)} total points, {len(stair_points)} stair points')
            
            # Store both point sets
            self.raw_points = np.array(raw_points, dtype=np.float32)
            self.stair_points = np.array(stair_points, dtype=np.float32)
            return self.stair_points  # Return stair points for backward compatibility
            
        except Exception as e:
            self.get_logger().error(f'Error loading ASCII PLY: {str(e)}')
            return None
            
    def load_binary_ply(self, filename, header_text):
        """Load binary format PLY"""
        import struct
        
        try:
            # Parse header for properties
            vertex_count = self.parse_vertex_count(header_text)
            if vertex_count == 0:
                return None
                
            # Parse property information
            properties = []
            lines = header_text.split('\n')
            for line in lines:
                if line.startswith('property'):
                    parts = line.split()
                    if len(parts) >= 3:
                        prop_type = parts[1]
                        prop_name = parts[2]
                        properties.append((prop_type, prop_name))
                        
                    self.get_logger().info(f'Found {vertex_count} vertices in binary PLY with properties: {properties}')
                    
                    # Check if normal vectors are available
                    has_normals = any(name in ['nx', 'ny', 'nz'] for _, name in properties)
                    if has_normals:
                        self.get_logger().info('✅ Normal vectors (nx, ny, nz) detected in PLY file')
                    else:
                        self.get_logger().warn('⚠️ No normal vectors found in PLY file')            # Calculate bytes per vertex
            bytes_per_vertex = 0
            format_string = '<'  # little endian
            prop_offsets = {}
            
            for prop_type, prop_name in properties:
                prop_offsets[prop_name] = bytes_per_vertex
                if prop_type in ['float', 'float32']:
                    format_string += 'f'
                    bytes_per_vertex += 4
                elif prop_type in ['double', 'float64']:
                    format_string += 'd'
                    bytes_per_vertex += 8
                elif prop_type in ['uchar', 'uint8']:
                    format_string += 'B'
                    bytes_per_vertex += 1
                else:
                    self.get_logger().warn(f'Unknown property type: {prop_type}')
                    format_string += 'f'  # default to float
                    bytes_per_vertex += 4
                    
            # Read binary data
            with open(filename, 'rb') as f:
                # Skip header
                while True:
                    line = f.readline().decode('utf-8', errors='ignore').strip()
                    if line == 'end_header':
                        break
                        
                # Read vertex data
                raw_points = []
                stair_points = []
                
                for i in range(vertex_count):
                    data = f.read(bytes_per_vertex)
                    if len(data) < bytes_per_vertex:
                        break
                        
                    values = struct.unpack(format_string, data)
                    
                    
                    # Extract x, y, z, nx, ny, nz (find their indices in properties)
                    x_idx = next((i for i, (_, name) in enumerate(properties) if name == 'x'), 0)
                    y_idx = next((i for i, (_, name) in enumerate(properties) if name == 'y'), 1)
                    z_idx = next((i for i, (_, name) in enumerate(properties) if name == 'z'), 2)
                    nx_idx = next((i for i, (_, name) in enumerate(properties) if name == 'nx'), None)
                    ny_idx = next((i for i, (_, name) in enumerate(properties) if name == 'ny'), None)
                    nz_idx = next((i for i, (_, name) in enumerate(properties) if name == 'nz'), None)
                    z_idx = next((i for i, (_, name) in enumerate(properties) if name == 'z'), 2)
                    
                    x = values[x_idx] if x_idx < len(values) else 0.0
                    y = values[y_idx] if y_idx < len(values) else 0.0  
                    z = values[z_idx] if z_idx < len(values) else 0.0
                    
                    # Extract normal vectors if available
                    nx = values[nx_idx] if nx_idx is not None and nx_idx < len(values) else 0.0
                    ny = values[ny_idx] if ny_idx is not None and ny_idx < len(values) else 0.0
                    nz = values[nz_idx] if nz_idx is not None and nz_idx < len(values) else 0.0
                    
                    # Add to raw points with normal vectors [x, y, z, intensity, nx, ny, nz]
                    raw_points.append([x, y, z, 128, nx, ny, nz])
                    
                    # For binary files, add all points as potential stairs
                    # (The AI model will do the actual stair detection)
                    intensity = 200  # Make all points visible for segmentation
                    stair_points.append([x, y, z, intensity, nx, ny, nz])
                    
            self.get_logger().info(f'Parsed {len(raw_points)} total points, {len(stair_points)} stair points')
            
            # Store both point sets
            self.raw_points = np.array(raw_points, dtype=np.float32)
            self.stair_points = np.array(stair_points, dtype=np.float32)
            return self.stair_points  # Return stair points for backward compatibility
            
        except Exception as e:
            self.get_logger().error(f'Error loading binary PLY: {str(e)}')
            return None
            
    def parse_vertex_count(self, header_text):
        """Parse vertex count from PLY header"""
        lines = header_text.split('\n')
        for line in lines:
            if 'element vertex' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'vertex' and i + 1 < len(parts):
                        return int(parts[i + 1])
        return 0
            
    def publish_clouds(self):
        """Publish both raw and occupied point clouds"""
        self.get_logger().debug(f'publish_clouds called (count: {self.publish_count})')
        
        if not hasattr(self, 'file_loaded') or not self.file_loaded or self.raw_points is None or self.stair_points is None:
            self.get_logger().debug('No PLY data loaded - skipping publish')
            return
        
        # Handle single-shot publishing when loop_playback is False
        if not self.loop_playback and self.publish_count > 0:
            self.get_logger().debug('Single-shot mode: already published, skipping')
            return
            
        # Create header with STABLE timestamp for consistent processing
        header = Header()
        # Use a deterministic timestamp instead of constantly changing time
        if not hasattr(self, 'stable_timestamp'):
            self.stable_timestamp = self.get_clock().now().to_msg()
        header.stamp = self.stable_timestamp
        header.frame_id = 'livox_frame'
            
        # Publish raw cloud
        self.publish_points(self.raw_points, header, self.pub_raw, "raw")
        
        self.publish_count += 1
        
        if not self.loop_playback:
            self.get_logger().info(f'Single-shot publish complete (count: {self.publish_count}). Node will keep running.')
        else:
            self.get_logger().info(f'Published clouds (count: {self.publish_count})')
        
        
    def publish_points(self, points, header, publisher, name):
        """Helper method to publish point cloud"""
        
        if points is None or len(points) == 0:
            self.get_logger().warn(f'No points to publish for {name} cloud')
            return
            
        # Create fields including normal vectors
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.UINT8, count=1),
            PointField(name='nx', offset=13, datatype=PointField.FLOAT32, count=1),
            PointField(name='ny', offset=17, datatype=PointField.FLOAT32, count=1),
            PointField(name='nz', offset=21, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Convert to bytes manually for proper format with normals
        point_step = 25  # 4+4+4+1+4+4+4 = 25 bytes per point
        row_step = point_step * len(points)
        
        data = bytearray()
        for point in points:
            if len(point) >= 7:  # New format with normals
                x, y, z, intensity, nx, ny, nz = point[:7]
            else:  # Old format without normals
                x, y, z, intensity = point[:4]
                nx, ny, nz = 0.0, 0.0, 0.0  # Default normals
                
            # Pack all fields as little endian
            import struct
            data.extend(struct.pack('<f', x))    # x
            data.extend(struct.pack('<f', y))    # y  
            data.extend(struct.pack('<f', z))    # z
            data.extend(struct.pack('<B', int(intensity)))  # intensity
            data.extend(struct.pack('<f', nx))   # nx
            data.extend(struct.pack('<f', ny))   # ny
            data.extend(struct.pack('<f', nz))   # nz
        
        # Create PointCloud2 message manually
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = point_step
        cloud_msg.row_step = row_step
        cloud_msg.data = bytes(data)
        cloud_msg.is_dense = False
        
        publisher.publish(cloud_msg)
        self.get_logger().info(f'Published {name} cloud with {len(points)} points')

def main(args=None):
    import signal
    import sys
    
    rclpy.init(args=args)
    
    # Handle Ctrl+C more aggressively
    def signal_handler(sig, frame):
        print('\nCtrl+C detected! Shutting down PLY publisher...')
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    node = None
    try:
        node = PLYPublisher()
        print('PLY Publisher started. Press Ctrl+C to stop.')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt caught!')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print('PLY Publisher shutdown complete.')

if __name__ == '__main__':
    main()
