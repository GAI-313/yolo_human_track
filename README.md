# YOLO Human Track ROS2パッケージ

YOLOv8とByteTrackを使用した人物姿勢推定と追跡のROS2パッケージ。RGB-Dカメラから人物の骨格情報を検出し、3D空間での位置を可視化します。

## 特徴

- YOLOv8-poseによる高精度な人物姿勢推定
- ByteTrackによるマルチオブジェクト追跡
- RGB-Dカメラとの統合（Intel Realsense D435対応）
- 3D空間での骨格可視化（RViz対応）
- カスタマイズ可能なパラメータ設定

## 環境構築

### 依存パッケージのインストール

```bash
# ROS2 Humble (Ubuntu 22.04推奨)
sudo apt install ros-humble-desktop

# Python依存関係
pip install ultralytics opencv-python numpy message-filters

# ROS2パッケージ
sudo apt install ros-humble-cv-bridge ros-humble-vision-msgs
```

### パッケージのビルド

1. ROS2ワークスペースに本パッケージをクローン
```bash
cd ~/ros2_ws/src
git clone https://github.com/your_repository/yolo_human_track.git
```

2. 依存関係の解決とビルド
```bash
cd ~/ros2_ws
rosdep install -i --from-path src --rosdistro humble -y
colcon build --packages-select yolo_human_track
source install/setup.bash
```

## 使用方法

### 起動方法

デフォルト設定で起動:
```bash
ros2 launch yolo_human_track yolo_human_track_launch.py
```

カスタムトピック指定で起動:
```bash
ros2 launch yolo_human_track yolo_human_track_launch.py \
  color_image:=/your_camera/color/image_raw \
  depth_image:=/your_camera/depth/image_rect_raw
```

## Launchファイル引数

| 引数名 | デフォルト値 | 説明 |
|--------|--------------|------|
| `color_image` | `/d435/color/image_raw` | RGBカメラ画像トピック |
| `depth_image` | `/d435/depth/image_rect_raw` | 深度画像トピック |
| `depth_camerainfo` | `/d435/depth/camera_info` | 深度カメラ情報トピック |
| `color_camerainfo` | `/d435/color/camera_info` | RGBカメラ情報トピック |
| `params_file` | `パッケージ内のyolo_human_track.yaml` | パラメータファイルパス |

## パラメータ説明

`params/yolo_human_track.yaml` で設定可能なパラメータ:

### YOLO姿勢推定ノード

| パラメータ | デフォルト値 | 説明 |
|------------|--------------|------|
| `detect_conf` | 0.5 | 人物検出の信頼度閾値 (0-1) |
| `detect_iou` | 0.5 | NMS(非最大抑制)のIOU閾値 (0-1) |
| `yolo_model` | "yolov8n-pose.pt" | 使用するYOLOモデルファイル名 |
| `track_thresh` | 0.5 | ByteTrackの高信頼度検出閾値 |
| `match_thresh` | 0.8 | ByteTrackのマッチングIOU閾値 |
| `track_buffer` | 30 | トラックを保持するフレーム数 |

### 3D可視化ノード

| パラメータ | デフォルト値 | 説明 |
|------------|--------------|------|
| `depth_scale` | 0.001 | 深度値のスケール係数 (mm→m変換) |
| `marker_lifetime` | 0.05 | RVizマーカーの表示時間(秒) |
| `depth_sample_size` | 5 | 深度サンプリング領域のサイズ (奇数推奨) |

## カスタマイズ

### モデルの変更

`yolo_model` パラメータで異なるYOLOv8-poseモデルを指定可能:
- yolov8n-pose.pt (ナノ, 最軽量)
- yolov8s-pose.pt (スモール)
- yolov8m-pose.pt (ミディアム)
- yolov8l-pose.pt (ラージ)
- yolov8x-pose.pt (エクストララージ)

### トピックの変更

異なるカメラを使用する場合、launchファイルの引数で対応するトピック名を指定:
```bash
ros2 launch yolo_human_track yolo_human_track_launch.py \
  color_image:=/other_camera/color/image_raw \
  depth_image:=/other_camera/aligned_depth_to_color/image_raw
```
