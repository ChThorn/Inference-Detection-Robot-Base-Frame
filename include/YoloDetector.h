#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>
#include "TransformMatrixCamToRobot.h"

struct GraspPose {
    Eigen::Vector3d position_cam;     // Position in camera frame (meters)
    Eigen::Vector3d position_robot;   // Position in robot base frame (meters)
    Eigen::Matrix3d orientation;      // Rotation matrix for gripper orientation
    cv::Rect bbox;                    // Object bounding box
    cv::Point grasp_point;            // Preferred grasp location in image
    float confidence;                 // Detection confidence (0-1)
    float object_angle;
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_path,
                 const std::string& config_path,
                 const std::string& intrinsic_file,
                 const std::string& extrinsic_file,
                 float confidence_thresh = 0.5,
                 float nms_thresh = 0.4,
                 cv::Size network_input_size = {416, 416},
                 float grasp_point_x_ratio = 1.1);

    cv::Mat detect(const cv::Mat& color_frame, const cv::Mat& depth_frame);
    const std::vector<GraspPose>& get_detections() const { return detections_; }

private:
    cv::dnn::Net net_;
    float conf_thresh_;
    float nms_thresh_;
    cv::Size network_input_size_;
    float grasp_point_x_ratio_;
    
    std::vector<GraspPose> detections_;
    TransformMatrixCamToRobot transformer_;

    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& network_outputs);
    std::vector<std::string> get_output_names() const;
};

#endif