// #include "YoloDetector.h"
// #include <opencv2/dnn.hpp>
// #include <iostream>

// YoloDetector::YoloDetector(const std::string& modelPath, 
//                            const std::string& configPath,
//                            float confidenceThreshold,
//                            float nmsThreshold)
//     : confThreshold(confidenceThreshold), nmsThreshold(nmsThreshold) {
//     try {
//         net = cv::dnn::readNetFromDarknet(configPath, modelPath);
//         if (net.empty()) {
//             throw std::runtime_error("Failed to create network");
//         }
//         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//     } catch (const std::exception& e) {
//         throw std::runtime_error("Failed to load the network: " + std::string(e.what()));
//     }
// }

// cv::Mat YoloDetector::detect(const cv::Mat& frame) {
//     cv::Mat blob, processedFrame = frame.clone();
//     try {
//         cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
//         net.setInput(blob);

//         std::vector<cv::Mat> outs;
//         net.forward(outs, getOutputsNames());

//         postprocess(processedFrame, outs);

//         return processedFrame;
//     } catch (const std::exception& e) {
//         std::cerr << "Error during detection: " << e.what() << std::endl;
//         return frame;
//     }
// }

// void YoloDetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
//     std::vector<float> confidences;
//     std::vector<cv::Rect> boxes;
//     std::vector<int> classIds;
//     detections.clear();

//     for (const auto& out : outs) {
//         float* data = (float*)out.data;
//         for (int j = 0; j < out.rows; ++j, data += out.cols) {
//             cv::Mat scores = out.row(j).colRange(5, out.cols);
//             cv::Point classIdPoint;
//             double maxClassScore;
            
//             // Get the highest class score and its index
//             cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
//             float confidence = maxClassScore * data[4];  // Combine with objectness score

//             if (confidence > confThreshold) {
//                 int centerX = (int)(data[0] * frame.cols);
//                 int centerY = (int)(data[1] * frame.rows);
//                 int width = (int)(data[2] * frame.cols);
//                 int height = (int)(data[3] * frame.rows);
//                 int left = centerX - width / 2;
//                 int top = centerY - height / 2;

//                 classIds.push_back(classIdPoint.x);
//                 confidences.push_back(confidence);
//                 boxes.push_back(cv::Rect(left, top, width, height));
//             }
//         }
//     }

//     std::vector<int> indices;
//     cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

//     for (size_t i = 0; i < indices.size(); ++i) {
//         int idx = indices[i];
//         cv::Rect box = boxes[idx];

//         cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
//         cv::Point graspPoint(center.x + (int)(0.60 * box.width), center.y);

//         Detection det;
//         det.box = box;
//         det.confidence = confidences[idx];
//         det.graspPoint = graspPoint;
//         detections.push_back(det);

//         cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 3);
//         cv::circle(frame, graspPoint, 5, cv::Scalar(0, 0, 255), -1);
//         cv::line(frame, center, graspPoint, cv::Scalar(255, 0, 0), 2);

//         std::string label = cv::format("Conf: %.2f", confidences[idx]);
//         int baseLine;
//         cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//         cv::rectangle(frame, cv::Point(box.x, box.y - labelSize.height),
//                       cv::Point(box.x + labelSize.width, box.y + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
//         cv::putText(frame, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
//     }
// }

// float YoloDetector::getAverageDepth(const cv::Mat& depth, const cv::Point& point, int windowSize) {
//     int halfWindow = windowSize / 2;
//     std::vector<float> validDepths;
    
//     for(int y = -halfWindow; y <= halfWindow; y++) {
//         for(int x = -halfWindow; x <= halfWindow; x++) {
//             int newY = point.y + y;
//             int newX = point.x + x;
            
//             if(newX >= 0 && newX < depth.cols && newY >= 0 && newY < depth.rows) {
//                 float d = depth.at<float>(newY, newX);
//                 if(d > 0.2 && d < 6.0) {  // Valid depth range
//                     validDepths.push_back(d);
//                 }
//             }
//         }
//     }
    
//     if(validDepths.empty()) return 0.0f;
    
//     // Sort and take median
//     std::sort(validDepths.begin(), validDepths.end());
//     return validDepths[validDepths.size() / 2];
// }

// float YoloDetector::calculateDepth(const cv::Mat& depth, const cv::Point& point) {
//     return getAverageDepth(depth, point);
// }

// std::vector<std::string> YoloDetector::getOutputsNames() {
//     static std::vector<std::string> names;
//     if (names.empty()) {
//         std::vector<int> outLayers = net.getUnconnectedOutLayers();
//         std::vector<std::string> layersNames = net.getLayerNames();
//         names.resize(outLayers.size());
//         for (size_t i = 0; i < outLayers.size(); ++i) {
//             names[i] = layersNames[outLayers[i] - 1];
//         }
//     }
//     return names;
// }



#include "YoloDetector.h"
#include <opencv2/core/eigen.hpp>
#include <algorithm>

namespace {
    bool load_matrix_from_yaml(const std::string& filename, 
                             const std::string& key,
                             cv::Mat& mat) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;
        fs[key] >> mat;
        return !mat.empty();
    }
}

YoloDetector::YoloDetector(const std::string& model_path,
                          const std::string& config_path,
                          const std::string& intrinsic_file,
                          const std::string& extrinsic_file,
                          float confidence_thresh,
                          float nms_thresh)
    : conf_thresh_(confidence_thresh), nms_thresh_(nms_thresh) {
    
    // Load network
    net_ = cv::dnn::readNetFromDarknet(config_path, model_path);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load camera parameters
    if (!load_matrix_from_yaml(intrinsic_file, "camera_matrix", camera_matrix_) ||
        !load_matrix_from_yaml(intrinsic_file, "distortion_coefficients", dist_coeffs_)) {
        throw std::runtime_error("Failed to load intrinsic parameters");
    }

    // Load extrinsic calibration
    cv::Mat extrinsic_cv;
    if (!load_matrix_from_yaml(extrinsic_file, "extrinsic_matrix", extrinsic_cv) ||
        extrinsic_cv.rows != 4 || extrinsic_cv.cols != 4) {
        throw std::runtime_error("Invalid extrinsic matrix");
    }
    cv::cv2eigen(extrinsic_cv, extrinsic_);
}

cv::Mat YoloDetector::detect(const cv::Mat& color_frame, const cv::Mat& depth_frame) {
    cv::Mat blob, processed_frame = color_frame.clone();
    
    try {
        // Network inference
        cv::dnn::blobFromImage(color_frame, blob, 1/255.0, {416,416}, cv::Scalar(), true, false);
        net_.setInput(blob);
        
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, get_output_names());

        // Process detections
        postprocess(processed_frame, outputs);

        // Calculate 3D positions for valid detections
        for (auto& detection : detections_) {
            // Get median depth in 5x5 window
            const float depth = calculate_depth(depth_frame, detection.grasp_point);
            
            if (depth >= 0.2 && depth <= 6.0) {
                // Convert to 3D coordinates
                detection.position_cam = pixel_to_camera(detection.grasp_point, depth);
                detection.position_robot = transform_to_robot(detection.position_cam);

                // Annotate frame
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

Eigen::Vector3d YoloDetector::pixel_to_camera(const cv::Point& pixel, float depth) const {
    std::vector<cv::Point2f> distorted{pixel};
    std::vector<cv::Point2f> undistorted;
    
    cv::undistortPoints(distorted, undistorted, camera_matrix_, dist_coeffs_);
    
    return Eigen::Vector3d(
        undistorted[0].x * depth,
        undistorted[0].y * depth,
        depth
    );
}

Eigen::Vector3d YoloDetector::transform_to_robot(const Eigen::Vector3d& point_cam) const {
    const Eigen::Vector4d cam_homog(point_cam.x(), point_cam.y(), point_cam.z(), 1.0);
    const Eigen::Vector4d robot_homog = extrinsic_ * cam_homog;
    return robot_homog.head<3>() / robot_homog.w();
}

float YoloDetector::calculate_depth(const cv::Mat& depth, const cv::Point& point) const {
    const int window = 2;  // 5x5 window
    std::vector<float> valid_depths;
    
    for (int dy = -window; dy <= window; ++dy) {
        for (int dx = -window; dx <= window; ++dx) {
            const cv::Point pt(point.x + dx, point.y + dy);
            if (pt.x >= 0 && pt.x < depth.cols && 
                pt.y >= 0 && pt.y < depth.rows) {
                const float d = depth.at<float>(pt);
                if (d > 0.2f && d < 6.0f) valid_depths.push_back(d);
            }
        }
    }
    
    if (valid_depths.empty()) return 0.0f;
    std::nth_element(valid_depths.begin(), 
                    valid_depths.begin() + valid_depths.size()/2,
                    valid_depths.end());
    return valid_depths[valid_depths.size()/2];
}

void YoloDetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outputs) {
    detections_.clear();
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    
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
                
                boxes.emplace_back(cx - width/2, cy - height/2, width, height);
                confidences.push_back(static_cast<float>(confidence));
            }
        }
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, indices);

    // Store valid detections
    for (const int idx : indices) {
        GraspPose pose;
        pose.bbox = boxes[idx];
        pose.confidence = confidences[idx];
        pose.grasp_point = cv::Point(
            pose.bbox.x + pose.bbox.width * 0.6,  // 60% from left edge
            pose.bbox.y + pose.bbox.height/2
        );
        
        cv::rectangle(frame, pose.bbox, {0,255,0}, 2);
        cv::circle(frame, pose.grasp_point, 5, {0,0,255}, -1);
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