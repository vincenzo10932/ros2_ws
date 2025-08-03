from setuptools import find_packages, setup

package_name = 'my_terrain_seg'

setup(
    name=package_name,
    version='0.1.0',
    # <-- flat layout: look here for packages
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'setuptools',
        'torch',
        'numpy',
        'scipy',
        'open3d',
    ],
    zip_safe=True,
    maintainer='vincent',
    maintainer_email='vincent@todo.todo',
    description='Point‑cloud terrain segmentation → OctoMap integration',
    license='Apache‑2.0',
    entry_points={
        'console_scripts': [
            'segmentation_node = my_terrain_seg.segmentation_and_octomap.segmentation_node:main',
            'octomap_node      = my_terrain_seg.segmentation_and_octomap.octomap_node:main',
            'real_octomap_node = my_terrain_seg.segmentation_and_octomap.real_octomap_node:main',
            'ply_publisher     = my_terrain_seg.segmentation_and_octomap.ply_publisher:main',
        ],
    },
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name + '/launch', [
            'launch/terrain_seg.launch.py',
            'launch/offline_test.launch.py',
            'launch/simple_offline_test.launch.py',
            'launch/configurable_ply_test.launch.py'
        ]),
    ],
)
