// #include <librealsense2/rs.hpp>
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include "YoloDetector.h"


// int main(int argc, char** argv) {
//     try {
//         // Initialize RealSense pipeline
//         rs2::pipeline pipe;
//         rs2::config cfg;
//         cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30); // RGB stream
//         cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);  // Depth stream
//         pipe.start(cfg);

//         // Initialize YOLO detector
//         std::string modelPath = "/home/thornch/Documents/Cpp/camera_preparation/detection_based_3d_individualObject/yolov3.weights";
//         std::string configPath = "/home/thornch/Documents/Cpp/camera_preparation/detection_based_3d_individualObject/yolov3.cfg";
//         YoloDetector detector(modelPath, configPath);

//         // Wait for a coherent pair of frames: depth and color
//         rs2::frameset frames = pipe.wait_for_frames();
//         rs2::frame color_frame = frames.get_color_frame();
//         rs2::depth_frame depth_frame = frames.get_depth_frame();

//         if (!color_frame || !depth_frame) {
//             std::cerr << "Error: Could not retrieve frames.\n";
//             return -1;
//         }

//         // Convert frames to OpenCV matrices
//         cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
//         cv::Mat depth(cv::Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
//         depth.convertTo(depth, CV_32F, 1.0 / 1000.0); // Convert millimeters to meters

//         // Perform object detection
//         cv::Mat result = detector.detect(color);

//         // Add depth visualization
//         cv::Mat depth_visualization;
//         depth.convertTo(depth_visualization, CV_8UC1, 255.0/3.0); // Scale for visibility (0-3m range)
//         cv::applyColorMap(depth_visualization, depth_visualization, cv::COLORMAP_JET);

//         // Access and process all detections
//         const auto& detections = detector.getDetections();
//         for (size_t i = 0; i < detections.size(); i++) {
//             cv::Point graspPoint = detections[i].graspPoint;

//             // Extract depth at grasp point
//             float depth_value = YoloDetector::calculateDepth(depth, graspPoint);
//             if (depth_value <= 0.2 || depth_value > 6.0) {
//                 std::cout << "Object " << i + 1 << ": Depth unavailable at grasp point (" 
//                           << graspPoint.x << ", " << graspPoint.y << ")\n";
//                 continue;
//             }

//             std::cout << "Object " << i + 1 << ":\n"
//                       << "  Confidence: " << detections[i].confidence << "\n"
//                       << "  Grasp point: (" << graspPoint.x << ", " << graspPoint.y << ")\n"
//                       << "  Depth: " << depth_value << " meters\n";
                      
//             // Draw depth value on the image
//             std::string depthText = cv::format("%.3fm", depth_value);
//             cv::putText(result, depthText, 
//                         cv::Point(graspPoint.x - 20, graspPoint.y - 20),
//                         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
//         }

//         // Display results
//         cv::imshow("Object Detection with Grasp Points", result);
//         cv::imshow("Depth Map", depth_visualization);
        
//         // Wait for a key press before closing
//         cv::waitKey(0);

//         // Stop the pipeline
//         pipe.stop();

//         return 0;
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
// }

// #include <librealsense2/rs.hpp>
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include "YoloDetector.h"

// int main(int argc, char** argv) {
//     try {
//         // Initialize RealSense
//         rs2::pipeline pipe;
//         rs2::config cfg;
//         cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
//         cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
//         pipe.start(cfg);
        
//         YoloDetector detector(
//             "/home/thornch/Documents/Cpp/camera_preparation/detection_based_3d_individualObject/yolov3.weights",
//             "/home/thornch/Documents/Cpp/camera_preparation/detection_based_3d_individualObject/yolov3.cfg",
//             "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/intrinsic_params.yml",
//             "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/extrinsic_params.yml"
//         );

//         // Capture frames
//         rs2::frameset frames = pipe.wait_for_frames();
//         auto color_frame = frames.get_color_frame();
//         auto depth_frame = frames.get_depth_frame();

//         // Convert to OpenCV format
//         cv::Mat color(cv::Size(640, 480), CV_8UC3, 
//                      (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
//         cv::Mat depth(cv::Size(640, 480), CV_16UC1, 
//                      (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
//         depth.convertTo(depth, CV_32F, 1.0 / 1000.0);

//         // Detect objects and get poses
//         cv::Mat result = detector.detect(color, depth);

//         // Process and display results
//         for (const auto& grasp : detector.get_detections()) {  // Changed from getDetections to get_detections
//             std::cout << "Object detected:\n"
//                       << "  Camera frame position (m): "
//                       << grasp.position_cam.transpose() << "\n"
//                       << "  Robot frame position (m): "
//                       << grasp.position_robot.transpose() << "\n"
//                       << "  Confidence: " << grasp.confidence << "\n\n";

//             // Visualize on image
//             cv::circle(result, grasp.grasp_point, 5, cv::Scalar(0, 0, 255), -1);  // Changed from graspPoint to grasp_point
//             std::string pos_text = cv::format("(%.2f, %.2f, %.2f)m", 
//                 grasp.position_robot.x(), 
//                 grasp.position_robot.y(), 
//                 grasp.position_robot.z());
//             cv::putText(result, pos_text, 
//                 cv::Point(grasp.grasp_point.x - 20, grasp.grasp_point.y - 20),  // Changed from graspPoint to grasp_point
//                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
//         }

//         // Display results
//         //------Visualize depth graph----
//         cv::Mat depth_vis;
//         depth.convertTo(depth_vis, CV_8UC1, 255.0/3.0); // Scale for 0-3m range
//         cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);
//         cv::imshow("Depth Map", depth_vis);

//         //------Visualize RGB graph-------
//         cv::imshow("Detection Results", result);
//         cv::waitKey(0);

//         pipe.stop();
//         return 0;

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
// }


#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "YoloDetector.h"

int main(int argc, char** argv) {
    try {
        // Initialize RealSense
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        pipe.start(cfg);
        
        // Create named windows
        cv::namedWindow("Detection Results", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth Map", cv::WINDOW_AUTOSIZE);
        
        YoloDetector detector(
            "/home/thornch/Documents/Cpp/camera_preparation/detection_based_3d_individualObject/yolov3.weights",
            "/home/thornch/Documents/Cpp/camera_preparation/detection_based_3d_individualObject/yolov3.cfg",
            "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/intrinsic_params.yml",
            "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/extrinsic_params.yml"
        );

        // Capture a single frame
        rs2::frameset frames = pipe.wait_for_frames();
        auto color_frame = frames.get_color_frame();
        auto depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) {
            throw std::runtime_error("Failed to capture frames");
        }

        // Convert to OpenCV format
        cv::Mat color(cv::Size(640, 480), CV_8UC3, 
                     (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(640, 480), CV_16UC1, 
                     (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        depth.convertTo(depth, CV_32F, 1.0 / 1000.0);

        // Create depth visualization
        cv::Mat depth_vis;
        depth.convertTo(depth_vis, CV_8UC1, 255.0/3.0);
        cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);

        // Detect objects and get poses
        cv::Mat result = detector.detect(color, depth);

        // Process detections
        for (const auto& grasp : detector.get_detections()) {
            std::cout << "Object detected:\n"
                      << "  Camera frame position (m): "
                      << grasp.position_cam.transpose() << "\n"
                      << "  Robot frame position (m): "
                      << grasp.position_robot.transpose() << "\n"
                      << "  Confidence: " << grasp.confidence << "\n\n";
        }

        // Display images and wait for key press
        cv::imshow("Detection Results", result);
        cv::imshow("Depth Map", depth_vis);
        cv::waitKey(0);  // Wait indefinitely until a key is pressed

        // Cleanup
        pipe.stop();
        cv::destroyAllWindows();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}