#!/usr/bin/env python3
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.node import Node
import rclpy

from human_pose_msgs.msg import Pose2DArray, Pose2D, RoiPose, RoiPoseArray
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

from ultralytics import YOLO

from cv_bridge import CvBridge
import numpy as np
import cv2
import time

from collections import deque
import traceback


SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7)
]

class BYTETracker:
    def __init__(self, 
                 track_thresh=0.5,
                 match_thresh=0.8,
                 track_buffer=30,
                 frame_rate=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.next_id = 1
        
    def update(self, detections, image):
        # 検出を信頼度で分類
        dets_high = [d for d in detections if d['conf'] > self.track_thresh]
        dets_low = [d for d in detections if d['conf'] <= self.track_thresh]
        
        # 既存トラックを非アクティブに設定
        for track in self.tracked_tracks:
            track['active'] = False
        
        # 高信頼度検出でマッチング
        matched_pairs = self._match_detections(self.tracked_tracks, dets_high)
        
        # 未マッチのトラックと検出を取得
        unmatched_tracks = [t for t in self.tracked_tracks if not t['active']]
        unmatched_dets = [d for i,d in enumerate(dets_high) if i not in [j for _,j in matched_pairs]]
        
        # 低信頼度検出で再マッチング
        rematched_pairs = self._match_detections(unmatched_tracks, dets_low, self.match_thresh)
        
        # 未マッチ検出を新規トラックとして追加
        for det in unmatched_dets:
            self._add_track(det)
            
        # トラック状態を更新
        self._update_track_states()
        
        # 結果をフォーマット
        tracks = []
        for track in self.tracked_tracks:
            if track['active']:
                tracks.append({
                    'bbox': track['bbox'],
                    'track_id': track['track_id'],
                    'conf': track['conf'],
                    'keypoints': track['keypoints']
                })
        
        return tracks
    
    def _match_detections(self, tracks, detections, thresh=0.5):
        if not tracks or not detections:
            return []
            
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i,j] = self._iou(track['bbox'], det['bbox'])
                
        matched_indices = np.where(iou_matrix >= thresh)
        matched_pairs = list(zip(matched_indices[0], matched_indices[1]))
        
        for i, j in matched_pairs:
            tracks[i]['bbox'] = detections[j]['bbox']
            tracks[i]['conf'] = detections[j]['conf']
            tracks[i]['keypoints'] = detections[j]['keypoints']
            tracks[i]['active'] = True
            
        return matched_pairs
    
    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)
    
    def _add_track(self, detection):
        new_track = {
            'bbox': detection['bbox'].tolist(),
            'track_id': self.next_id,
            'conf': detection['conf'],
            'keypoints': detection['keypoints'],
            'active': True,
            'frames': 0
        }
        self.tracked_tracks.append(new_track)
        self.next_id += 1
    
    def _update_track_states(self):
        to_remove = []
        
        for i, track in enumerate(self.tracked_tracks):
            if track['active']:
                track['frames'] += 1
            else:
                if track['frames'] > self.track_buffer:
                    to_remove.append(i)
                else:
                    track['frames'] += 1
        
        for i in sorted(to_remove, reverse=True):
            if i < len(self.tracked_tracks):
                self.removed_tracks.append(self.tracked_tracks[i])
                del self.tracked_tracks[i]


class PoseEstimate(Node):
    def __init__(self):
        super().__init__('yolo_human_track_main')

        self.declare_parameter('auto_bringup', False)
        self.declare_parameter('detect_conf', 0.5)
        self.declare_parameter('detect_iou', 0.5)
        self.declare_parameter('yolo_model', 'yolov8n-pose.pt')
        self.declare_parameter('track_thresh', 0.5)
        self.declare_parameter('match_thresh', 0.8)
        self.declare_parameter('track_buffer', 30)

        self.execute = self.get_parameter('auto_bringup').value
        self.detect_conf = self.get_parameter('detect_conf').value
        self.detect_iou = self.get_parameter('detect_iou').value
        self.yolo_model = self.get_parameter('yolo_model').value
        track_thresh = self.get_parameter('track_thresh').value
        match_thresh = self.get_parameter('match_thresh').value
        track_buffer = self.get_parameter('track_buffer').value

        self.model = None
        if self.execute:
            self.model = self.bringup_model()

        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer
        )
        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.pose_pub = self.create_publisher(
            Pose2DArray,
            'yolo_human_track/pose/poses_2d',
            10
        )
        
        self.rgb_sub = self.create_subscription(
            Image,
            '/arm_camera/color/image_raw',
            self.rgb_callback,
            qos_profile=qos_profile
        )
        self.srv = self.create_service(
            SetBool,
            'yolo_human_track/execute',
            self._cb_execute_manager
        )
        
        self.get_logger().info("PoseEstimate node initialized")

    def bringup_model(self):
        self.get_logger().info(f"Loading model: {self.yolo_model}")
        return YOLO(self.yolo_model)

    def _cb_execute_manager(self, req: SetBool.Request, res: SetBool.Response):
        if req.data:
            try:
                self.model = self.bringup_model()
                self.execute = True
                res.message = f"Loaded {self.yolo_model}"
                self.get_logger().info(res.message)
            except Exception as e:
                res.success = False
                res.message = f"Model loading failed: {str(e)}"
                self.get_logger().error(res.message)
                return res
        else:
            self.execute = False
            self.model = None
            res.message = 'Stop detection'
            self.get_logger().info(res.message)
        
        res.success = True
        return res

    def rgb_callback(self, msg):
        if not self.execute or self.model is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            #self.get_logger().info('processing')
            
            # YOLO推論
            results = self.model(cv_image, verbose=False, conf=self.detect_conf, iou=self.detect_iou)
            
            # 検出結果をフォーマット
            detections = []
            for result in results:
                if result.keypoints is None:
                    continue
                    
                for box, kpts in zip(result.boxes, result.keypoints):
                    keypoints = kpts.xy[0].cpu().numpy()
                    conf = float(box.conf.item()) if box.conf is not None else 0.0
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': bbox,
                        'keypoints': keypoints,
                        'conf': conf
                    })
            
            # トラッキング更新
            tracks = self.tracker.update(detections, cv_image)
            
            # 結果可視化
            debug_image = cv_image.copy()
            for track in tracks:
                self._visualize_track(debug_image, track)
            
            # 結果表示
            cv2.imshow("Human Pose Tracking", debug_image)
            cv2.waitKey(1)
            
            # ROSメッセージ発行
            pose_array = Pose2DArray()
            for track in tracks:
                pose = Pose2D()
                pose.id = track['track_id']
                pose.conf = track.get('conf', 0.0)
                
                if 'keypoints' in track and track['keypoints'] is not None:
                    pose.keypoints = self._create_roi_array(track['keypoints'])
                else:
                    pose.keypoints = RoiPoseArray()
                
                pose_array.poses.append(pose)
            
            pose_array.header = msg.header
            self.pose_pub.publish(pose_array)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            traceback.print_exc()

    def _create_roi_array(self, keypoints):
        roi_array = RoiPoseArray()
        
        for i in range(17):
            roi = RoiPose()
            if i < len(keypoints):
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                roi.x = x
                roi.y = y
                roi.conf = 1.0 if x > 0 and y > 0 else 0.0
            else:
                roi.x = 0
                roi.y = 0
                roi.conf = 0.0
            
            roi_array.keypoints[i] = roi
        
        return roi_array

    def _visualize_track(self, image, track):
        bbox = track['bbox']
        track_id = track['track_id']
        
        # バウンディングボックス描画
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0), 2
        )
        
        # トラックID表示
        cv2.putText(
            image,
            f"ID: {track_id}",
            (int(bbox[0]), int(bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
        
        # キーポイントと骨格描画
        if 'keypoints' in track and track['keypoints'] is not None:
            keypoints = track['keypoints']
            
            # キーポイント描画
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                if x > 0 and y > 0:
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            
            # 骨格接続線描画
            for i, j in SKELETON:
                if i-1 < len(keypoints) and j-1 < len(keypoints):
                    start = (int(keypoints[i-1][0]), int(keypoints[i-1][1]))
                    end = (int(keypoints[j-1][0]), int(keypoints[j-1][1]))
                    if start[0] > 0 and start[1] > 0 and end[0] > 0 and end[1] > 0:
                        cv2.line(image, start, end, (255, 0, 0), 2)

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
