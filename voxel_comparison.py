#!/usr/bin/env python3
"""
Voxel Density Comparison Calculator
Shows how different parameters affect voxel count
"""

def calculate_voxel_impact():
    # Original configuration
    old_resolution = 0.1  # 10cm voxels
    old_volume_per_voxel = old_resolution ** 3
    
    # New configuration
    new_resolution = 0.05  # 5cm voxels
    new_volume_per_voxel = new_resolution ** 3
    
    # Calculate voxel density improvement
    density_multiplier = old_volume_per_voxel / new_volume_per_voxel
    
    print("🎯 VOXEL DENSITY COMPARISON")
    print("=" * 50)
    print(f"📊 OLD Configuration:")
    print(f"   Resolution: {old_resolution}m ({old_resolution*100}cm voxels)")
    print(f"   Volume per voxel: {old_volume_per_voxel:.6f} m³")
    print(f"   Previous result: ~80,344 total voxels")
    print()
    print(f"🚀 NEW Configuration:")
    print(f"   Resolution: {new_resolution}m ({new_resolution*100}cm voxels)")  
    print(f"   Volume per voxel: {new_volume_per_voxel:.6f} m³")
    print(f"   Expected multiplier: {density_multiplier:.1f}x more voxels")
    print(f"   Expected total: ~{80344 * density_multiplier:,.0f} voxels")
    print()
    
    # Show spatial coverage improvement
    print("📏 SPATIAL COVERAGE:")
    print(f"   Each dimension: {2}x more voxels")
    print(f"   Total 3D space: {2}³ = {density_multiplier:.0f}x more voxels")
    print()
    
    # Show other improvements
    print("⚡ OTHER IMPROVEMENTS:")
    print("   • Elevated sensor (Z=2.0m) → More raytracing coverage")
    print("   • Higher hit probability (0.8 vs 0.7) → More sensitive detection")
    print("   • Lower miss probability (0.2 vs 0.3) → Retains more voxels")
    print("   • Lower minimum threshold (0.05 vs 0.12) → Keeps more free space")
    print()
    
    print("🎯 EXPECTED RESULTS:")
    print(f"   Occupied voxels: ~{2675 * density_multiplier:,.0f} (vs {2675:,})")
    print(f"   Free voxels: ~{77669 * density_multiplier:,.0f} (vs {77669:,})")
    print(f"   Total voxels: ~{80344 * density_multiplier:,.0f} (vs {80344:,})")

if __name__ == "__main__":
    calculate_voxel_impact()
