#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


DEFAULT_COLOR_TOPIC = '/d435/color/image_raw'
DEFAULT_DEPTH_TOPIC = '/d435/depth/image_rect_raw'
DEFAULT_DEPTH_INFO = '/d435/depth/camera_info'
DEFAULT_COLOR_INFO = '/d435/color/camera_info'
DEFAULT_PARAMS_FILE = os.path.join(
    get_package_share_directory('yolo_human_track'),
    'params', 'yolo_human_track.yaml'
)


def generate_launch_description():

    ld = LaunchDescription()

    declare_color_image = DeclareLaunchArgument(
        'color_image', 
        default_value=DEFAULT_COLOR_TOPIC,
        description='RGB カメラトピック'
    )
    declare_depth_image = DeclareLaunchArgument(
        'depth_image',
        default_value=DEFAULT_DEPTH_TOPIC,
        description='深度画像トピック名'
    )
    
    declare_depth_info = DeclareLaunchArgument(
        'depth_camerainfo',
        default_value=DEFAULT_DEPTH_INFO,
        description='深度カメラ情報トピック名'
    )
    
    declare_color_info = DeclareLaunchArgument(
        'color_camerainfo',
        default_value=DEFAULT_COLOR_INFO,
        description='RGBカメラ情報トピック名'
    )
    
    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=DEFAULT_PARAMS_FILE,
        description='パラメータファイルのパス'
    )

    ld.add_action(declare_color_image)
    ld.add_action(declare_depth_image)
    ld.add_action(declare_depth_info)
    ld.add_action(declare_color_info)
    ld.add_action(declare_params_file)


    node_yolo_human_track = Node(
        package='yolo_human_track',
        executable='yolo_human_track',
        remappings=[
            ('color_image_raw', LaunchConfiguration('color_image'))
        ],
        parameters=[LaunchConfiguration('params_file')]
    )

    node_pose_transformer = Node(
        package='yolo_human_track',
        executable='pose_transformer',
        remappings=[
            ('depth_image_raw', LaunchConfiguration('depth_image')),
            ('depth_camerainfo', LaunchConfiguration('depth_camerainfo')),
            ('color_camerainfo', LaunchConfiguration('color_camerainfo'))
        ],
        parameters=[LaunchConfiguration('params_file')]
    )

    ld.add_action(node_yolo_human_track)
    ld.add_action(node_pose_transformer)


    return ld