#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <human_pose_msgs/msg/pose2_d_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <unordered_map>
#include <set>
#include <vector>
#include <algorithm>

using namespace std::chrono_literals;

class Pose3DVisualizer : public rclcpp::Node
{
public:
    Pose3DVisualizer() : Node("pose_3d_visualizer"), marker_id_(0), alignment_ready_(false)
    {
        // パラメータ宣言
        this->declare_parameter("depth_scale", 0.001);
        this->declare_parameter("marker_lifetime", 0.05);
        this->declare_parameter("depth_sample_size", 5);

        // QoS設定
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
        qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        qos.durability(rclcpp::DurabilityPolicy::Volatile);

        // サブスクライバー作成
        pose_sub_.subscribe(this, "/yolo_human_track/pose/poses_2d", qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, "/d435/depth/image_rect_raw", qos.get_rmw_qos_profile());
        depth_info_sub_.subscribe(this, "/d435/depth/camera_info", qos.get_rmw_qos_profile());
        color_info_sub_.subscribe(this, "/d435/color/camera_info", qos.get_rmw_qos_profile());

        // タイムシンクロナイザー
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), pose_sub_, depth_sub_, depth_info_sub_, color_info_sub_);
        sync_->registerCallback(std::bind(&Pose3DVisualizer::syncCallback, this,
            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

        // パブリッシャー
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/human_pose_markers", 10);

        RCLCPP_INFO(this->get_logger(), "Pose3DVisualizer initialized");
    }

private:
    // キーポイント接続情報
    const std::vector<std::pair<int, int>> KEYPOINT_CONNECTIONS = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {6, 8}, {7, 9},
        {8, 10}, {9, 11}, {12, 13},
        {11, 12}, {11, 13}, {12, 14}, {13, 15},
        {14, 16}, {15, 17}
    };

    void initializeCameraParameters(
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& depth_info,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& color_info)
    {
        depth_camera_matrix_ = cv::Mat(3, 3, CV_64F, const_cast<double*>(depth_info->k.data())).clone();
        color_camera_matrix_ = cv::Mat(3, 3, CV_64F, const_cast<double*>(color_info->k.data())).clone();
        depth_dist_coeffs_ = cv::Mat(1, depth_info->d.size(), CV_64F, const_cast<double*>(depth_info->d.data())).clone();
        color_dist_coeffs_ = cv::Mat(1, color_info->d.size(), CV_64F, const_cast<double*>(color_info->d.data())).clone();
        
        RCLCPP_INFO(this->get_logger(), "Camera parameters initialized");
        alignment_ready_ = true;
    }

    void syncCallback(
        const human_pose_msgs::msg::Pose2DArray::ConstSharedPtr pose_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr depth_msg,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr depth_info_msg,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr color_info_msg)
    {
        try {
            if (!alignment_ready_) {
                initializeCameraParameters(depth_info_msg, color_info_msg);
            }

            cv_bridge::CvImagePtr cv_depth;
            try {
                cv_depth = cv_bridge::toCvCopy(depth_msg, "16UC1");
            } catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            double depth_scale = this->get_parameter("depth_scale").as_double();
            double marker_lifetime = this->get_parameter("marker_lifetime").as_double();
            int sample_size = this->get_parameter("depth_sample_size").as_int();

            visualization_msgs::msg::MarkerArray marker_array;
            std::set<int> current_marker_ids;

            // 前回のマーカーを削除
            for (int marker_id : last_marker_ids_) {
                visualization_msgs::msg::Marker marker;
                marker.header = pose_msg->header;
                marker.id = marker_id;
                marker.action = visualization_msgs::msg::Marker::DELETE;
                marker_array.markers.push_back(marker);
            }

            for (const auto& pose : pose_msg->poses) {
                std::vector<cv::Point3d> valid_points;
                for (size_t i = 0; i < pose.keypoints.keypoints.size(); ++i) {
                    const auto& kp = pose.keypoints.keypoints[i];
                    if (kp.x > 0 && kp.y > 0) {
                        auto [u_depth, v_depth] = alignColorToDepth(
                            kp.x, kp.y, color_camera_matrix_, depth_camera_matrix_);
                        
                        double depth = getMedianDepth(
                            cv_depth->image, static_cast<int>(u_depth), 
                            static_cast<int>(v_depth), sample_size) * depth_scale;
                        
                        if (depth > 0) {
                            double x = (u_depth - depth_camera_matrix_.at<double>(0, 2)) * 
                                      depth / depth_camera_matrix_.at<double>(0, 0);
                            double y = (v_depth - depth_camera_matrix_.at<double>(1, 2)) * 
                                      depth / depth_camera_matrix_.at<double>(1, 1);
                            valid_points.emplace_back(x, y, depth);
                        }
                    }
                }

                if (valid_points.size() < 3) continue;

                cv::Point3d center = std::accumulate(
                    valid_points.begin(), valid_points.end(), cv::Point3d(0, 0, 0)) / 
                    static_cast<double>(valid_points.size());
                
                auto color = getPersonColor(pose.id);

                // 重心マーカー
                marker_id_++;
                current_marker_ids.insert(marker_id_);
                auto marker = createSphereMarker(
                    pose_msg->header, marker_id_, pose.id, "person",
                    {center.x, center.y, center.z}, {0.1, 0.1, 0.1},
                    {color[0], color[1], color[2], 1.0}, marker_lifetime);
                marker_array.markers.push_back(marker);

                // 骨格マーカー
                if (valid_points.size() >= 2) {
                    marker_id_++;
                    current_marker_ids.insert(marker_id_);
                    auto line_marker = createLineListMarker(
                        pose_msg->header, marker_id_, pose.id, "skeleton",
                        valid_points, {0.02, 0.0, 0.0},
                        {color[0], color[1], color[2], 0.8}, marker_lifetime);
                    marker_array.markers.push_back(line_marker);
                }

                // 手先マーカー
                const int LEFT_HAND_IDX = 9;
                const int RIGHT_HAND_IDX = 10;
                
                for (int hand_idx : {LEFT_HAND_IDX, RIGHT_HAND_IDX}) {
                    if (hand_idx < static_cast<int>(valid_points.size())) {
                        const auto& hand_pt = valid_points[hand_idx];
                        if (hand_pt.z > 0) {
                            std::vector<float> inverted_color = {
                                1.0f - color[0],
                                1.0f - color[1],
                                1.0f - color[2]
                            };
                            
                            marker_id_++;
                            current_marker_ids.insert(marker_id_);
                            auto hand_marker = createSphereMarker(
                                pose_msg->header, marker_id_, pose.id, 
                                "hand_" + std::to_string(hand_idx),
                                {hand_pt.x, hand_pt.y, hand_pt.z}, {0.08, 0.08, 0.08},
                                {inverted_color[0], inverted_color[1], inverted_color[2], 0.9},
                                marker_lifetime);
                            marker_array.markers.push_back(hand_marker);
                        }
                    }
                }
            }

            last_marker_ids_ = current_marker_ids;

            if (!marker_array.markers.empty()) {
                marker_pub_->publish(marker_array);
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000, "Processing error: %s", e.what());
        }
    }

    std::pair<double, double> alignColorToDepth(
        double u_color, double v_color, 
        const cv::Mat& color_matrix, const cv::Mat& depth_matrix)
    {
        double scale_x = depth_matrix.at<double>(0, 0) / color_matrix.at<double>(0, 0);
        double scale_y = depth_matrix.at<double>(1, 1) / color_matrix.at<double>(1, 1);
        
        double u_depth = (u_color - color_matrix.at<double>(0, 2)) * scale_x + depth_matrix.at<double>(0, 2);
        double v_depth = (v_color - color_matrix.at<double>(1, 2)) * scale_y + depth_matrix.at<double>(1, 2);
        
        return {u_depth, v_depth};
    }

    double getMedianDepth(const cv::Mat& depth_image, int u, int v, int size)
    {
        int half = size / 2;
        int u_min = std::max(0, u - half);
        int u_max = std::min(depth_image.cols - 1, u + half);
        int v_min = std::max(0, v - half);
        int v_max = std::min(depth_image.rows - 1, v + half);
        
        cv::Mat roi = depth_image(cv::Range(v_min, v_max + 1), cv::Range(u_min, u_max + 1));
        cv::Mat valid_depths = roi.clone().reshape(1);
        valid_depths = valid_depths(valid_depths > 0);
        
        if (valid_depths.empty()) return 0.0;
        
        cv::Mat sorted;
        cv::sort(valid_depths, sorted, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
        
        double q1 = sorted.at<uint16_t>(sorted.rows * 0.25);
        double q3 = sorted.at<uint16_t>(sorted.rows * 0.75);
        double iqr = q3 - q1;
        double lower_bound = q1 - 1.5 * iqr;
        double upper_bound = q3 + 1.5 * iqr;
        
        cv::Mat mask = (sorted >= lower_bound) & (sorted <= upper_bound);
        cv::Mat filtered = sorted.clone();
        filtered.setTo(0, ~mask);
        filtered = filtered(filtered > 0);
        
        if (!filtered.empty()) {
            return filtered.at<uint16_t>(filtered.rows / 2);
        } else {
            return sorted.at<uint16_t>(sorted.rows / 2);
        }
    }

    std::vector<float> getPersonColor(int person_id)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.2f, 1.0f);
        
        if (person_colors_.find(person_id) == person_colors_.end()) {
            person_colors_[person_id] = {
                dis(gen),
                dis(gen),
                dis(gen)
            };
        }
        return person_colors_[person_id];
    }

    visualization_msgs::msg::Marker createSphereMarker(
        const std_msgs::msg::Header& header,
        int marker_id, int person_id, const std::string& ns,
        const std::vector<double>& position,
        const std::vector<double>& scale,
        const std::vector<float>& color,
        double lifetime)
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "person_" + std::to_string(person_id) + "_" + ns;
        marker.id = marker_id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.lifetime = rclcpp::Duration::from_seconds(lifetime);
        marker.pose.position.x = position[0];
        marker.pose.position.y = position[1];
        marker.pose.position.z = position[2];
        marker.scale.x = scale[0];
        marker.scale.y = scale[1];
        marker.scale.z = scale[2];
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];
        marker.color.a = color[3];
        return marker;
    }

    visualization_msgs::msg::Marker createLineListMarker(
        const std_msgs::msg::Header& header,
        int marker_id, int person_id, const std::string& ns,
        const std::vector<cv::Point3d>& points,
        const std::vector<double>& scale,
        const std::vector<float>& color,
        double lifetime)
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "person_" + std::to_string(person_id) + "_" + ns;
        marker.id = marker_id;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.lifetime = rclcpp::Duration::from_seconds(lifetime);
        marker.scale.x = scale[0];
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];
        marker.color.a = color[3];

        for (const auto& [start_idx, end_idx] : KEYPOINT_CONNECTIONS) {
            if (start_idx < static_cast<int>(points.size()) && 
                end_idx < static_cast<int>(points.size())) {
                const auto& start_pt = points[start_idx];
                const auto& end_pt = points[end_idx];
                if (start_pt.z > 0 && end_pt.z > 0) {
                    geometry_msgs::msg::Point p1, p2;
                    p1.x = start_pt.x; p1.y = start_pt.y; p1.z = start_pt.z;
                    p2.x = end_pt.x; p2.y = end_pt.y; p2.z = end_pt.z;
                    marker.points.push_back(p1);
                    marker.points.push_back(p2);
                }
            }
        }
        return marker;
    }

    // メンバ変数
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        human_pose_msgs::msg::Pose2DArray,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::CameraInfo,
        sensor_msgs::msg::CameraInfo>;

    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    message_filters::Subscriber<human_pose_msgs::msg::Pose2DArray> pose_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> depth_info_sub_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> color_info_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    cv::Mat depth_camera_matrix_;
    cv::Mat color_camera_matrix_;
    cv::Mat depth_dist_coeffs_;
    cv::Mat color_dist_coeffs_;

    int marker_id_;
    std::unordered_map<int, std::vector<float>> person_colors_;
    std::set<int> last_marker_ids_;
    bool alignment_ready_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Pose3DVisualizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
