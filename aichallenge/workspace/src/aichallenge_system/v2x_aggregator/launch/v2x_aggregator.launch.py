from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    stale = LaunchConfiguration('stale_timeout_s')
    period = LaunchConfiguration('publish_period_s')

    return LaunchDescription([
        DeclareLaunchArgument('stale_timeout_s', default_value='1.0'),
        DeclareLaunchArgument('publish_period_s', default_value='0.05'),
        Node(
            package='v2x_aggregator',
            executable='v2x_aggregator_node',
            name='v2x_aggregator',
            output='screen',
            parameters=[{
                'stale_timeout_s': stale,
                'publish_period_s': period,
            }],
        ),
    ])
