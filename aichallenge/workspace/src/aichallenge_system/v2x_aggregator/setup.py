from setuptools import setup

package_name = 'v2x_aggregator'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/v2x_aggregator.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taikitanaka3',
    maintainer_email='taiki.tanaka@tier4.jp',
    description='V2X PointStamped aggregator -> V2XVehiclePositionArray',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'v2x_aggregator_node = v2x_aggregator.aggregator_node:main',
        ],
    },
)
