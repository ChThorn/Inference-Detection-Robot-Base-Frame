// #ifndef YOLO_DETECTOR_H
// #define YOLO_DETECTOR_H

// #include <opencv2/opencv.hpp>
// #include <vector>

// // Structure to store detection results including grasp points
// struct Detection {
//     cv::Rect box;
//     float confidence;
//     cv::Point graspPoint;
// };

// class YoloDetector {
// private:
//     cv::dnn::Net net;
//     float confThreshold;
//     float nmsThreshold;

//     std::vector<Detection> detections; // Store detections for external access

//     void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
//     std::vector<std::string> getOutputsNames();

//     // Add this new function declaration
//     static float getAverageDepth(const cv::Mat& depth, const cv::Point& point, int windowSize = 5);

// public:
//     YoloDetector(const std::string& modelPath, 
//                  const std::string& configPath,
//                  float confidenceThreshold = 0.5,
//                  float nmsThreshold = 0.4);

//     cv::Mat detect(const cv::Mat& frame);
//     const std::vector<Detection>& getDetections() const { return detections; }

//     // Add this public method to access depth calculation
//     static float calculateDepth(const cv::Mat& depth, const cv::Point& point);
// };

// #endif // YOLO_DETECTOR_H


#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>

struct GraspPose {
    Eigen::Vector3d position_cam;     // Position in camera frame (meters)
    Eigen::Vector3d position_robot;   // Position in robot base frame (meters)
    cv::Rect bbox;                    // Object bounding box
    cv::Point grasp_point;            // Preferred grasp location in image
    float confidence;                 // Detection confidence (0-1)
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_path,
                 const std::string& config_path,
                 const std::string& intrinsic_file,
                 const std::string& extrinsic_file,
                 float confidence_thresh = 0.5,
                 float nms_thresh = 0.4);

    cv::Mat detect(const cv::Mat& color_frame, const cv::Mat& depth_frame);
    const std::vector<GraspPose>& get_detections() const { return detections_; }

private:
    // Network and parameters
    cv::dnn::Net net_;
    float conf_thresh_;
    float nms_thresh_;
    
    // Camera parameters
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    Eigen::Matrix4d extrinsic_;

    // Detection storage
    std::vector<GraspPose> detections_;

    // Processing methods
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& network_outputs);
    Eigen::Vector3d pixel_to_camera(const cv::Point& pixel, float depth) const;
    Eigen::Vector3d transform_to_robot(const Eigen::Vector3d& point_cam) const;
    float calculate_depth(const cv::Mat& depth, const cv::Point& point) const;
    std::vector<std::string> get_output_names() const;
};

#endif // YOLO_DETECTOR_H