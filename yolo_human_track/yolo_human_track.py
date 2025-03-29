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

from collections import deque
import traceback


# ==============================================
# 骨格接続情報 (COCOキーポイントフォーマットに準拠)
# 各タプルは接続するキーポイントのインデックスを表す
# インデックスは1-based (YOLOv8の出力形式に合わせる)
# ==============================================
SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),  # 下半身
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),          # 上半身
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),          # 腕と頭部
    (2, 4), (3, 5), (4, 6), (5, 7)                     # 肩と腰の接続
]

class BYTETracker:
    """
    ByteTrackベースのオブジェクトトラッカー
    特徴:
    - 高信頼度検出と低信頼度検出を組み合わせた追跡
    - IOU(Intersection over Union)ベースのマッチング
    - トラックバッファによる一時的な消失対応
    
    """
    
    def __init__(self, 
                 track_thresh=0.5,
                 match_thresh=0.8,
                 track_buffer=30,
                 frame_rate=30):
        """
        Args:
            track_thresh: 高信頼度検出の閾値 (0-1)
            match_thresh: マッチングのIOU閾値 (0-1)
            track_buffer: トラックを保持するフレーム数
            frame_rate: 入力フレームレート (未使用だが将来の拡張用)
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        
        # トラック管理用リスト
        self.tracked_tracks = []    # アクティブなトラック
        self.lost_tracks = []       # 一時的に消失したトラック
        self.removed_tracks = []    # 削除されたトラック(デバッグ用)
        self.next_id = 1            # 次のトラックID
        
    def update(self, detections, image):
        """
        トラッキングのメイン更新処理
        
        Args:
            detections: 検出結果リスト [{'bbox': [x1,y1,x2,y2], 'conf': score, 'keypoints': [...]}]
            image: 入力画像 (未使用だがインターフェース統一のため保持)
            
        Returns:
            更新されたトラック情報リスト
        """
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
                    'keypoints': track['keypoints']  # キーポイント情報を保持
                })
        
        return tracks
    
    def _match_detections(self, tracks, detections, thresh=0.5):
        """
        IOUベースの検出マッチング
        
        Args:
            tracks: 既存トラックリスト
            detections: 検出結果リスト
            thresh: IOU閾値
            
        Returns:
            マッチした(トラックインデックス, 検出インデックス)のペアリスト
        """
        if not tracks or not detections:
            return []
            
        # IOUマトリックスを計算
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i,j] = self._iou(track['bbox'], det['bbox'])
                
        # 閾値以上のマッチを取得
        matched_indices = np.where(iou_matrix >= thresh)
        matched_pairs = list(zip(matched_indices[0], matched_indices[1]))
        
        # マッチしたトラックを更新
        for i, j in matched_pairs:
            tracks[i]['bbox'] = detections[j]['bbox']
            tracks[i]['conf'] = detections[j]['conf']
            tracks[i]['keypoints'] = detections[j]['keypoints']  # キーポイント更新
            tracks[i]['active'] = True
            
        return matched_pairs
    
    def _iou(self, box1, box2):
        """
        IOU(Intersection over Union)計算
        
        Args:
            box1: [x1,y1,x2,y2]形式のバウンディングボックス
            box2: [x1,y1,x2,y2]形式のバウンディングボックス
            
        Returns:
            IOU値 (0-1)
        """
        # 交差領域の計算
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)
    
    def _add_track(self, detection):
        """
        新規トラックを追加
        
        Args:
            detection: 検出結果 {'bbox': [...], 'conf': ..., 'keypoints': [...]}
        """
        new_track = {
            'bbox': detection['bbox'].tolist(),  # NumPy配列→リスト変換
            'track_id': self.next_id,
            'conf': detection['conf'],
            'keypoints': detection['keypoints'],  # キーポイント保持
            'active': True,
            'frames': 0  # トラックが継続したフレーム数
        }
        self.tracked_tracks.append(new_track)
        self.next_id += 1
    
    def _update_track_states(self):
        """
        トラック状態を更新:
        - アクティブなトラック: フレームカウンタを増加
        - 非アクティブなトラック: 
          - バッファサイズを超えたら削除
          - そうでなければフレームカウンタを増加
        """
        to_remove = []
        
        for i, track in enumerate(self.tracked_tracks):
            if track['active']:
                track['frames'] += 1
            else:
                if track['frames'] > self.track_buffer:
                    to_remove.append(i)
                else:
                    track['frames'] += 1
        
        # 逆順で削除 (インデックスがずれないように)
        for i in sorted(to_remove, reverse=True):
            if i < len(self.tracked_tracks):
                self.removed_tracks.append(self.tracked_tracks[i])  # デバッグ用に保持
                del self.tracked_tracks[i]


class PoseEstimate(Node):
    """
    メインノード
    """

    def __init__(self):
        super().__init__('yolo_human_track_main')

        # ==============================================
        # パラメータ設定と読み込み
        # ==============================================
        self.declare_parameter('auto_bringup', False)   # プログラム実行時に自動的に物体検出をスタート
        self.declare_parameter('detect_conf', 0.5)      # 人物検出の信頼度閾値
        self.declare_parameter('detect_iou', 0.5)       # NMSのIOU閾値
        self.declare_parameter('yolo_model', 'yolov8n-pose.pt')  # モデルファイル
        self.declare_parameter('track_thresh', 0.5)     # トラッキングの信頼度閾値
        self.declare_parameter('match_thresh', 0.8)     # マッチングのIOU閾値
        self.declare_parameter('track_buffer', 30)      # トラック保持フレーム数

        # パラメータ値を読み込み
        self.execute = self.get_parameter('auto_bringup').get_parameter_value().bool_array_value
        self.detect_conf = self.get_parameter('detect_conf').get_parameter_value().double_value
        self.detect_iou = self.get_parameter('detect_iou').get_parameter_value().double_value
        self.yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        track_thresh = self.get_parameter('track_thresh').get_parameter_value().double_value
        match_thresh = self.get_parameter('match_thresh').get_parameter_value().double_value
        track_buffer = self.get_parameter('track_buffer').get_parameter_value().integer_value

        # ==============================================
        # モデルとトラッカーの初期化
        # ==============================================
        if self.execute: self.model = self.bringup_model()  # YOLOv8-poseモデル
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer
        )
        self.bridge = CvBridge()  # ROS-OpenCV変換
        
        # ==============================================
        # QoS設定 (カメラ画像用)
        # ==============================================
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # ベストエフォート
            durability=QoSDurabilityPolicy.VOLATILE,       # 非永続
            depth=10                                      # キューサイズ
        )
        
        # ==============================================
        # パブリッシャー/サブスクライバー/サービスの設定
        # ==============================================
        self.pose_pub = self.create_publisher(
            Pose2DArray,
            'yolo_human_track/pose/poses_2d',
            10
        )
        
        self.rgb_sub = self.create_subscription(
            Image,
            '/color_image_raw',
            self.rgb_callback,
            qos_profile=qos_profile
        )
        self.srv = self.create_service(
            SetBool,
            'yolo_human_track/execute',
            self._cb_execute_manager
        )
        
        self.get_logger().info("PoseEstimate node initialized (ByteTrack with skeleton visualization)")
    
    def bringup_model(self):
        self.get_logger().info("""
LOAD MODEL : %s
MINIMUM_CONF : %d                               
        """%(self.yolo_model, self.detect_conf))

        return YOLO(self.yolo_model) 
    

    def _cb_execute_manager(self, req: SetBool.Request, res: SetBool.Response):
        if req.data:
            self.model = self.bringup_model()
            self.execute = True
            res.message = "Load %s"%self.yolo_model
        
        else:
            self.execute = False
            del self.model
            res.message = 'Stop detection'
        
        res.success = True
        return res


    def rgb_callback(self, msg):
        """
        RGB画像コールバック関数

        """
        if self.execute:
            try:
                # ROS→OpenCV画像変換
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                
                # YOLOv8で推論 (信頼度閾値とNMS閾値を適用)
                results = self.model(cv_image, verbose=False, conf=self.detect_conf, iou=self.detect_iou)
                
                # 検出結果をフォーマット
                detections = self._create_detections(results)
                
                # ByteTrackで追跡
                tracks = self.tracker.update(detections, cv_image)
                
                # 結果をパブリッシュ
                self._publish_results(tracks, msg)
                
            except Exception as e:
                self.get_logger().error(f'Error processing image: {str(e)}')
                traceback.print_exc()

    def _create_detections(self, results):
        """
        YOLOの結果から検出情報を抽出

        """
        detections = []
        
        for result in results:
            if result.keypoints is None:  # 姿勢キーポイントがない場合はスキップ
                continue
                
            for box, kpts in zip(result.boxes, result.keypoints):
                # キーポイント座標 (NumPy配列)
                keypoints = kpts.xy[0].cpu().numpy()
                
                # 信頼度 (検出信頼度)
                conf = float(box.conf.item()) if box.conf is not None else 0.0
                
                # バウンディングボックス [x1,y1,x2,y2]
                bbox = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'conf': conf
                })
        
        return detections

    def _publish_results(self, tracks, msg):
        """
        トラッキング結果をROSメッセージとしてパブリッシュ
        
        Args:
            tracks: トラッキング結果
            msg: 元の画像メッセージ (ヘッダー情報用)
        """
        pose_array = Pose2DArray()
        debug_image = None  # 可視化用画像
        
        for track in tracks:
            pose = Pose2D()
            pose.id = track['track_id']
            pose.conf = track.get('conf', 0.0)
            
            # キーポイント情報がある場合のみ追加
            if 'keypoints' in track and track['keypoints'] is not None:
                pose.keypoints = self._create_roi_array(track['keypoints'])
            else:
                pose.keypoints = RoiPoseArray()  # 空データ
            
            # 最初のトラックで画像を初期化
            if debug_image is None:
                debug_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # トラック情報を画像に描画
            self._visualize_track(debug_image, track)
            pose_array.poses.append(pose)
        
        # 可視化画像を表示
        if debug_image is not None:
            cv2.imshow("Human Pose Tracking", debug_image)
            cv2.waitKey(1)
        
        # ROSメッセージをパブリッシュ
        pose_array.header = msg.header
        self.pose_pub.publish(pose_array)

    def _create_roi_array(self, keypoints):
        """
        ROSメッセージ用にキーポイント情報をフォーマット
        
        Args:
            keypoints: キーポイント座標リスト
            
        Returns:
            RoiPoseArrayメッセージ
        """
        roi_array = RoiPoseArray()
        
        for i in range(17):  # COCOフォーマットは17キーポイント
            roi = RoiPose()
            if i < len(keypoints):
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                roi.x = x
                roi.y = y
                # 有効な座標かどうかで信頼度を設定
                roi.conf = 1.0 if x > 0 and y > 0 else 0.0
            else:
                roi.x = 0
                roi.y = 0
                roi.conf = 0.0
            
            roi_array.keypoints[i] = roi
        
        return roi_array

    def _visualize_track(self, image, track):
        """
        トラッキング結果を画像に描画
        
        """
        # バウンディングボックス情報
        bbox = track['bbox']
        track_id = track['track_id']
        
        # バウンディングボックス描画
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0), 2  # 緑色
        )
        
        # トラックID表示
        cv2.putText(
            image,
            f"ID: {track_id}",
            (int(bbox[0]), int(bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2  # 緑色
        )
        
        # キーポイントと骨格がある場合のみ描画
        if 'keypoints' in track and track['keypoints'] is not None:
            keypoints = track['keypoints']
            
            # キーポイント描画 (赤色の円)
            for i, kp in enumerate(keypoints):
                x, y = int(kp[0]), int(kp[1])
                if x > 0 and y > 0:  # 有効な座標のみ描画
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 赤色
            
            # 骨格接続線描画 (青色の線)
            for i, j in SKELETON:
                # インデックスチェックと有効座標チェック
                if (i-1 < len(keypoints) and j-1 < len(keypoints) and
                    keypoints[i-1][0] > 0 and keypoints[i-1][1] > 0 and
                    keypoints[j-1][0] > 0 and keypoints[j-1][1] > 0):
                    
                    start = (int(keypoints[i-1][0]), int(keypoints[i-1][1]))
                    end = (int(keypoints[j-1][0]), int(keypoints[j-1][1]))
                    cv2.line(image, start, end, (255, 0, 0), 2)  # 青色


def main(args=None):
    """
    メイン関数
    ROS2ノードの初期化と実行
    """
    rclpy.init(args=args)
    node = PoseEstimate()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # クリーンアップ
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()