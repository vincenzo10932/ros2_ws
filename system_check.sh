#!/bin/bash
# Quick test to verify the 3-terminal setup works

echo "=== QUICK SYSTEM TEST ==="
echo

echo "1. Checking Python environment..."
source .venv/bin/activate
python3 -c "import torch, numpy; print('✓ PyTorch and NumPy available')"

echo "2. Checking ROS2 setup..."
source install/setup.bash
ros2 pkg list | grep my_terrain_seg > /dev/null && echo "✓ my_terrain_seg package found"

echo "3. Checking executables..."
ls install/my_terrain_seg/lib/my_terrain_seg/ | while read exe; do
    echo "  ✓ $exe"
done

echo "4. Checking RViz2..."
which rviz2 > /dev/null && echo "✓ RViz2 available at $(which rviz2)"

echo "5. Checking PLY data..."
ls data/ply/*.ply | while read ply; do
    echo "  ✓ $(basename $ply)"
done

echo "6. Testing PLY publisher (5 second test)..."
timeout 5s ros2 run my_terrain_seg ply_publisher &
PUB_PID=$!
sleep 2
echo "   Checking for /cloud_raw topic..."
ros2 topic list | grep cloud_raw > /dev/null && echo "   ✓ /cloud_raw topic found" || echo "   ✗ /cloud_raw topic NOT found"
kill $PUB_PID 2>/dev/null || true

echo
echo "=== SYSTEM CHECK COMPLETE ==="
echo "Follow terminal_setup_guide.txt for the 3-terminal setup"
echo "If /cloud_raw topic not found, check that PLY publisher is running in Terminal 1"
