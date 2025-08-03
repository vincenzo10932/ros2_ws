#!/bin/bash
# ROS2 Environment Setup Script for External Terminals
# Usage: source ./setup_ros2.sh

echo "🚀 Setting up ROS2 environment..."

# Check if we're in the right directory
if [ ! -d "install" ]; then
    echo "❌ install directory not found!"
    echo "Please run this from /home/vincent/ros2_ws"
    return 1
fi

# Source ROS2 base environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "📦 Sourcing ROS2 Humble..."
    source /opt/ros/humble/setup.bash
else
    echo "❌ ROS2 Humble not found at /opt/ros/humble/setup.bash"
    return 1
fi

# Source workspace
if [ -f "install/setup.bash" ]; then
    echo "🔧 Sourcing workspace..."
    source install/setup.bash
else
    echo "❌ Workspace not built! Run: colcon build --packages-select my_terrain_seg"
    return 1
fi

# Verify setup
echo "✅ Checking environment..."
if command -v ros2 >/dev/null 2>&1; then
    echo "✅ ROS2 command available"
else
    echo "❌ ROS2 command not found"
    return 1
fi

if ros2 pkg list | grep -q my_terrain_seg; then
    echo "✅ my_terrain_seg package found"
else
    echo "❌ my_terrain_seg package not found"
    return 1
fi

echo ""
echo "🎯 Environment ready! You can now run:"
echo "   ros2 launch my_terrain_seg configurable_ply_test.launch.py"
echo "   ros2 run my_terrain_seg ply_publisher"
echo ""
echo "💡 Common commands:"
echo "   # Test PLY publisher:"
echo "   ros2 run my_terrain_seg ply_publisher --ros-args -p ply_file:=/home/vincent/ros2_ws/data/ply/Stair_1.ply"
echo ""
echo "   # Run full pipeline:"
echo "   ros2 launch my_terrain_seg configurable_ply_test.launch.py"
echo ""
echo "   # Check topics:"
echo "   ros2 topic list | grep cloud"
