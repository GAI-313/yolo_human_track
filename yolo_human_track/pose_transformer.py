#!/usr/bin/env python3
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.node import Node
import rclpy

from message_filters import ApproximateTimeSynchronizer, Subscriber
from human_pose_msgs.msg import Pose2DArray
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Pose, PoseArray, Quaternion, PoseStamped
from std_msgs.msg import ColorRGBA
from yolo_world_interfaces.msg import PoseStampedArray

from cv_bridge import CvBridge
import numpy as np
import random

import tf_transformations

import traceback


class Pose3DVisualizer(Node):
    KEYPOINT_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9),
        (8, 10), (9, 11), (12, 13),
        (11, 12), (11, 13), (12, 14), (13, 15),
        (14, 16), (15, 17)
    ]

    def __init__(self):
        super().__init__('pose_3d_visualizer')
        
        # パラメータ設定
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('marker_lifetime', 0.05)
        self.declare_parameter('depth_sample_size', 5)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        
        pose_sub = Subscriber(self, Pose2DArray, '/yolo_human_track/pose/poses_2d')
        depth_sub = Subscriber(self, Image, 'depth_image_raw')
        depth_info_sub = Subscriber(self, CameraInfo, 'depth_camerainfo')
        color_info_sub = Subscriber(self, CameraInfo, 'color_camerainfo')
        
        self.ts = ApproximateTimeSynchronizer(
            [pose_sub, depth_sub, depth_info_sub, color_info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.marker_pub = self.create_publisher(MarkerArray, '/yolo_human_track/markers', 10)
        self.pose_stamped_array_pub = self.create_publisher(PoseStampedArray, '/yolo_human_track/pose/poses_3d', 10)
        
        self.bridge = CvBridge()
        self.depth_camera_matrix = None
        self.color_camera_matrix = None
        self.depth_dist_coeffs = None
        self.color_dist_coeffs = None
        self.marker_id = 0
        self.person_colors = {}
        self.last_marker_ids = set()
        self.alignment_ready = False

    def initialize_camera_parameters(self, depth_info, color_info):
        """カメラパラメータを初期化"""
        self.depth_camera_matrix = np.array(depth_info.k).reshape(3, 3)
        self.color_camera_matrix = np.array(color_info.k).reshape(3, 3)
        self.depth_dist_coeffs = np.array(depth_info.d)
        self.color_dist_coeffs = np.array(color_info.d)
        self.get_logger().info("カメラパラメータ初期化完了")

    def _calculate_orientation(self, keypoints_3d):
        """キーポイントから向きを計算"""
        try:
            # 肩と腰の中心点を計算
            left_shoulder = np.array(keypoints_3d[5])
            right_shoulder = np.array(keypoints_3d[6])
            left_hip = np.array(keypoints_3d[11])
            right_hip = np.array(keypoints_3d[12])
            
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            
            # 体の中心軸 (頭方向が+z)
            body_axis = shoulder_center - hip_center
            body_axis[2] = 0  # 水平面のみ考慮
            body_axis = body_axis / np.linalg.norm(body_axis)
            
            # 前方ベクトル (体の向き)
            forward_vec = np.array([body_axis[0], body_axis[1], 0])
            
            # 右方向ベクトル (x軸)
            right_vec = np.cross(np.array([0, 0, 1]), forward_vec)
            right_vec = right_vec / np.linalg.norm(right_vec)
            
            # 上方向ベクトル (z軸)
            up_vec = np.array([0, 0, 1])
            
            # 回転行列を作成
            rotation_matrix = np.column_stack([right_vec, forward_vec, up_vec])
            
            # クォータニオンに変換
            quat = tf_transformations.quaternion_from_matrix(
                np.vstack([np.column_stack([rotation_matrix, [0, 0, 0]]), [[0, 0, 0, 1]]])
            )
            
            return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            
        except Exception as e:
            self.get_logger().warn(f"向き計算エラー: {str(e)}")
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    def sync_callback(self, pose_msg, depth_msg, depth_info_msg, color_info_msg):
        try:
            if not self.alignment_ready:
                self.initialize_camera_parameters(depth_info_msg, color_info_msg)
                self.alignment_ready = True
            
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            depth_scale = self.get_parameter('depth_scale').value
            marker_lifetime = self.get_parameter('marker_lifetime').value
            sample_size = self.get_parameter('depth_sample_size').value
            
            marker_array = MarkerArray()
            current_marker_ids = set()
            
            # PoseStampedArrayを作成
            pose_stamped_array = PoseStampedArray()
            pose_stamped_array.header = depth_msg.header  # カメラフレームを使用
            pose_stamped_array.header.frame_id = depth_msg.header.frame_id  # カメラフレームIDを設定
            
            for pose in pose_msg.poses:
                valid_points = []
                keypoints_3d = []
                
                # キーポイントを3D座標に変換
                for kp in pose.keypoints.keypoints:
                    if kp.x > 0 and kp.y > 0:
                        u_depth, v_depth = self._align_color_to_depth(
                            kp.x, kp.y, 
                            self.color_camera_matrix, 
                            self.depth_camera_matrix
                        )
                        
                        depth = self._get_median_depth(
                            depth_image, 
                            int(u_depth), 
                            int(v_depth), 
                            sample_size
                        ) * depth_scale
                        
                        if depth > 0:
                            x = (u_depth - self.depth_camera_matrix[0, 2]) * depth / self.depth_camera_matrix[0, 0]
                            y = (v_depth - self.depth_camera_matrix[1, 2]) * depth / self.depth_camera_matrix[1, 1]
                            valid_points.append((x, y, depth))
                            keypoints_3d.append(np.array([x, y, depth]))
                
                if len(valid_points) < 5:  # 最低5つのキーポイントが必要
                    continue
                
                # 重心計算
                center = np.mean(valid_points, axis=0)
                color = self._get_person_color(pose.id)
                
                # 姿勢情報を作成
                person_pose = Pose()
                person_pose.position.x = center[0]
                person_pose.position.y = center[1]
                person_pose.position.z = center[2]
                
                # 向きを計算
                if len(keypoints_3d) >= 13:  # 必要なキーポイントがある場合
                    try:
                        # 肩と腰のキーポイントを取得
                        left_shoulder = keypoints_3d[5]
                        right_shoulder = keypoints_3d[6]
                        left_hip = keypoints_3d[11]
                        right_hip = keypoints_3d[12]
                        
                        # 体の向きを計算
                        shoulder_center = (left_shoulder + right_shoulder) / 2
                        hip_center = (left_hip + right_hip) / 2
                        body_dir = shoulder_center - hip_center
                        body_dir[2] = 0  # 水平成分のみ
                        body_dir_norm = body_dir / np.linalg.norm(body_dir)
                        
                        # 前方ベクトル (y軸)
                        forward_vec = body_dir_norm
                        
                        # 右方向ベクトル (x軸)
                        right_vec = np.cross(np.array([0, 0, 1]), forward_vec)
                        right_vec = right_vec / np.linalg.norm(right_vec)
                        
                        # 上方向ベクトル (z軸)
                        up_vec = np.array([0, 0, 1])
                        
                        # 回転行列を作成
                        rot_matrix = np.column_stack([right_vec, forward_vec, up_vec])
                        
                        # クォータニオンに変換
                        quat = tf_transformations.quaternion_from_matrix(
                            np.vstack([np.column_stack([rot_matrix, [0, 0, 0]]), [0, 0, 0, 1]]))
                        
                        person_pose.orientation = Quaternion(
                            x=quat[0], y=quat[1], z=quat[2], w=quat[3])
                        
                    except Exception as e:
                        self.get_logger().warn(f"向き計算エラー: {str(e)}")
                        person_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                else:
                    person_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                
                # PoseStampedを作成して配列に追加
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = depth_msg.header.stamp  # タイムスタンプは同期
                pose_stamped.header.frame_id = f"person_{pose.id}"  # 人物IDをframe_idとして設定
                pose_stamped.pose = person_pose
                
                pose_stamped_array.posestampedarray.append(pose_stamped)
                
                # マーカー作成 (重心)
                self.marker_id += 1
                current_marker_ids.add(self.marker_id)
                marker = Marker()
                marker.header = pose_msg.header
                marker.ns = f"person_{pose.id}"
                marker.id = self.marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.lifetime = rclpy.duration.Duration(seconds=marker_lifetime).to_msg()
                marker.pose.position = Point(x=center[0], y=center[1], z=center[2])
                marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
                marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
                marker_array.markers.append(marker)
                
                # 向きを示す矢印マーカー
                self.marker_id += 1
                current_marker_ids.add(self.marker_id)
                arrow_marker = Marker()
                arrow_marker.header = pose_msg.header
                arrow_marker.ns = f"direction_{pose.id}"
                arrow_marker.id = self.marker_id
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                arrow_marker.lifetime = rclpy.duration.Duration(seconds=marker_lifetime).to_msg()
                arrow_marker.pose = person_pose  # 姿勢情報をそのまま使用
                arrow_marker.scale = Vector3(x=0.2, y=0.05, z=0.05)  # サイズ調整
                arrow_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)  # 黄色
                marker_array.markers.append(arrow_marker)
                
                # 骨格マーカー
                if len(valid_points) >= 2:
                    self.marker_id += 1
                    current_marker_ids.add(self.marker_id)
                    line_marker = Marker()
                    line_marker.header = pose_msg.header
                    line_marker.ns = f"skeleton_{pose.id}"
                    line_marker.id = self.marker_id
                    line_marker.type = Marker.LINE_LIST
                    line_marker.action = Marker.ADD
                    line_marker.lifetime = rclpy.duration.Duration(seconds=marker_lifetime).to_msg()
                    line_marker.scale.x = 0.02
                    line_marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.8)
                    
                    for (start_idx, end_idx) in self.KEYPOINT_CONNECTIONS:
                        if start_idx < len(valid_points) and end_idx < len(valid_points):
                            start_pt = valid_points[start_idx]
                            end_pt = valid_points[end_idx]
                            if start_pt[2] > 0 and end_pt[2] > 0:
                                line_marker.points.append(Point(
                                    x=start_pt[0], y=start_pt[1], z=start_pt[2]))
                                line_marker.points.append(Point(
                                    x=end_pt[0], y=end_pt[1], z=end_pt[2]))
                    
                    marker_array.markers.append(line_marker)
                
                # 手先マーカー
                LEFT_HAND_IDX = 9
                RIGHT_HAND_IDX = 10
                
                for hand_idx in [LEFT_HAND_IDX, RIGHT_HAND_IDX]:
                    if hand_idx < len(valid_points):
                        hand_pt = valid_points[hand_idx]
                        if hand_pt[2] > 0:
                            inverted_color = (
                                1.0 - color[0],
                                1.0 - color[1],
                                1.0 - color[2]
                            )
                            
                            self.marker_id += 1
                            current_marker_ids.add(self.marker_id)
                            hand_marker = Marker()
                            hand_marker.header = pose_msg.header
                            hand_marker.ns = f"hand_{pose.id}_{hand_idx}"
                            hand_marker.id = self.marker_id
                            hand_marker.type = Marker.SPHERE
                            hand_marker.action = Marker.ADD
                            hand_marker.lifetime = rclpy.duration.Duration(seconds=marker_lifetime).to_msg()
                            hand_marker.pose.position = Point(
                                x=hand_pt[0],
                                y=hand_pt[1],
                                z=hand_pt[2]
                            )
                            hand_marker.scale = Vector3(x=0.08, y=0.08, z=0.08)
                            hand_marker.color = ColorRGBA(
                                r=inverted_color[0],
                                g=inverted_color[1],
                                b=inverted_color[2],
                                a=0.9
                            )
                            marker_array.markers.append(hand_marker)
            
            self.last_marker_ids = current_marker_ids
            
            # マーカーをパブリッシュ
            if marker_array.markers:
                self.marker_pub.publish(marker_array)
            
            # PoseStampedArrayをパブリッシュ
            if len(pose_stamped_array.posestampedarray) > 0:
                self.pose_stamped_array_pub.publish(pose_stamped_array)
                
        except Exception as e:
            self.get_logger().error(f"処理エラー: {str(e)}", throttle_duration_sec=1)
            traceback.print_exc()

    def _align_color_to_depth(self, u_color, v_color, color_matrix, depth_matrix):
        """RGB座標を深度画像座標に変換"""
        scale_x = depth_matrix[0, 0] / color_matrix[0, 0]
        scale_y = depth_matrix[1, 1] / color_matrix[1, 1]
        
        u_depth = (u_color - color_matrix[0, 2]) * scale_x + depth_matrix[0, 2]
        v_depth = (v_color - color_matrix[1, 2]) * scale_y + depth_matrix[1, 2]
        
        return u_depth, v_depth

    def _get_median_depth(self, depth_image, u, v, size):
        """周辺領域の深度中央値を取得"""
        h, w = depth_image.shape
        half = size // 2
        
        u_min = max(0, u-half)
        u_max = min(w, u+half+1)
        v_min = max(0, v-half)
        v_max = min(h, v+half+1)
        
        roi = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = roi[roi > 0]
        
        if valid_depths.size == 0:
            return 0.0
        
        q1 = np.percentile(valid_depths, 25)
        q3 = np.percentile(valid_depths, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_depths = valid_depths[(valid_depths >= lower_bound) & (valid_depths <= upper_bound)]
        
        return np.median(filtered_depths) if filtered_depths.size > 0 else np.median(valid_depths)

    def _get_person_color(self, person_id):
        """人物ごとに一意の色を生成"""
        if person_id not in self.person_colors:
            self.person_colors[person_id] = (
                random.uniform(0.2, 1.0),
                random.uniform(0.2, 1.0),
                random.uniform(0.2, 1.0)
            )
        return self.person_colors[person_id]

def main(args=None):
    rclpy.init(args=args)
    node = Pose3DVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
