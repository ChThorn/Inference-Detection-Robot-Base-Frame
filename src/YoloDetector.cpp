#include "YoloDetector.h"
#include <opencv2/dnn.hpp>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <cmath>

YoloDetector::YoloDetector(
    const std::string& model_path,
    const std::string& config_path,
    const std::string& intrinsic_file,
    const std::string& extrinsic_file,
    float confidence_thresh,
    float nms_thresh,
    cv::Size network_input_size,
    float grasp_point_x_ratio)
    : conf_thresh_(confidence_thresh), 
      nms_thresh_(nms_thresh),
      network_input_size_(network_input_size),
      grasp_point_x_ratio_(grasp_point_x_ratio),
      transformer_(intrinsic_file, extrinsic_file) {
    
    // Load network
    net_ = cv::dnn::readNetFromDarknet(config_path, model_path);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load YOLO model from " + model_path);
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

cv::Mat YoloDetector::detect(const cv::Mat& color_frame, const cv::Mat& depth_frame) {
    cv::Mat blob, processed_frame = color_frame.clone();
    
    try {
        // Preprocess
        cv::dnn::blobFromImage(color_frame, blob, 1/255.0, network_input_size_, cv::Scalar(), true, false);
        net_.setInput(blob);
        
        // Forward pass
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, get_output_names());

        // Postprocess detections and annotate frame
        postprocess(processed_frame, outputs);

        // Calculate 3D positions
        for (auto& detection : detections_) {
            float depth = transformer_.calculateDepth(depth_frame, detection.grasp_point);
            
            detection.position_cam = Eigen::Vector3d::Zero();
            detection.position_robot = Eigen::Vector3d::Zero();
            
            if (!std::isnan(depth)) {
                detection.position_cam = transformer_.pixelToCamera(detection.grasp_point, depth);
                detection.position_robot = transformer_.transformToRobot(detection.position_cam);

                // Annotate robot frame position on the image
                std::ostringstream oss;
                oss << "[" << std::fixed << std::setprecision(2)
                    << detection.position_robot.x() << ", "
                    << detection.position_robot.y() << ", "
                    << detection.position_robot.z() << "]";
                
                cv::putText(processed_frame, oss.str(),
                           detection.grasp_point + cv::Point(-50, -20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,255}, 2);
            }
        }

        return processed_frame;

    } catch (const std::exception& e) {
        std::cerr << "Detection error: " << e.what() << std::endl;
        return color_frame.clone();
    }
}

void YoloDetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outputs) {
    detections_.clear();
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    
    // Parse the outputs from the network
    for (const auto& output : outputs) {
        auto* data = (float*)output.data;
        for (int i = 0; i < output.rows; ++i, data += output.cols) {
            cv::Mat scores = output.row(i).colRange(5, output.cols);
            cv::Point class_id;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id);
            
            if (confidence > conf_thresh_) {
                const float cx = data[0] * frame.cols;
                const float cy = data[1] * frame.rows;
                const float width = data[2] * frame.cols;
                const float height = data[3] * frame.rows;
                
                boxes.emplace_back(cx - width / 2, cy - height / 2, width, height);
                confidences.push_back(static_cast<float>(confidence));
            }
        }
    }
    
    // Perform Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, indices);

    // Store final detections
    for (const int idx : indices) {
        GraspPose pose;
        pose.bbox = boxes[idx];
        
        // Compute grasp point using ratio from constructor
        int grasp_x = pose.bbox.x + static_cast<int>(pose.bbox.width * grasp_point_x_ratio_);
        int grasp_y = pose.bbox.y + pose.bbox.height / 2;
        pose.grasp_point = cv::Point(grasp_x, grasp_y);

        // Use a fixed orientation for a top-down grasp
        float fixed_angle = 0.0f; // Fixed angle (radians) for all objects
        pose.object_angle = fixed_angle;
        
        // Eigen::Matrix3d grip_orientation;
        // grip_orientation << cos(fixed_angle), -sin(fixed_angle), 0,
        //                     sin(fixed_angle),  cos(fixed_angle), 0,
        //                     0,                0,               -1;

        // Updated for top-down grasp (Z downward in robot frame):
        Eigen::Matrix3d grip_orientation;
        grip_orientation << 1,  0,  0,
                            0, -1,  0,  // Flip Y-axis if needed (depends on robot's frame)
                            0,  0, -1;  // Z-axis points downward
        pose.orientation = grip_orientation;
        pose.orientation = grip_orientation;
        
        pose.confidence = confidences[idx];
        
        // Draw bounding box, grasp point, and orientation arrow
        cv::rectangle(frame, pose.bbox, cv::Scalar(0, 255, 0), 2);
        cv::circle(frame, pose.grasp_point, 5, cv::Scalar(0, 0, 255), -1);
        
        cv::Point center(pose.bbox.x + pose.bbox.width / 2, 
                         pose.bbox.y + pose.bbox.height / 2);
        cv::Point direction(50 * cos(fixed_angle), 50 * sin(fixed_angle));
        cv::arrowedLine(frame, center, center + direction, 
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        
        detections_.push_back(pose);
    }
}

std::vector<std::string> YoloDetector::get_output_names() const {
    std::vector<std::string> names;
    const std::vector<int> out_layers = net_.getUnconnectedOutLayers();
    const std::vector<std::string> layers = net_.getLayerNames();
    
    names.reserve(out_layers.size());
    for (const int idx : out_layers) {
        names.push_back(layers[idx - 1]);
    }
    return names;
}
