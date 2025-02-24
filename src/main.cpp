#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "YoloDetector.h"

int main(int argc, char** argv) {
    try {
        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        pipe.start(cfg);

        // Align depth to color
        rs2::align align_to_color(RS2_STREAM_COLOR);

        // Warmup camera
        for (int i = 0; i < 30; i++) pipe.wait_for_frames();

        // Capture and align a single frame
        auto frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);
        auto color_frame = frames.get_color_frame();
        auto depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) {
            throw std::runtime_error("Failed to capture frames");
        }

        // Convert to OpenCV matrices
        cv::Mat color(cv::Size(640, 480), CV_8UC3, 
                     (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(640, 480), CV_16UC1, 
                     (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        
        // Convert depth to meters
        depth.convertTo(depth, CV_32F, 0.001f);

        // Initialize detector
        YoloDetector detector(
            "/home/thornch/Documents/Cpp/camera_preparation/Inference-Detection-Robot-Base-Frame/yolov3.weights",
            "/home/thornch/Documents/Cpp/camera_preparation/Inference-Detection-Robot-Base-Frame/yolov3.cfg",
            "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/intrinsic_params.yml",
            "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/extrinsic_params.yml"
        );
        // Detect objects
        cv::Mat result = detector.detect(color, depth);

        // Process and display results
        for (const auto& grasp : detector.get_detections()) {
            if (std::isnan(grasp.position_robot.x())) continue;  // Skip invalid

            std::cout << "Object detected:\n"
                     << "  Camera frame position (m): "
                     << grasp.position_cam.transpose() << "\n"
                     << "  Robot frame position (m): "
                     << grasp.position_robot.transpose() << "\n"
                     << "  Confidence: " << grasp.confidence << "\n"
                     << "  Orientation matrix:\n" 
                     << grasp.orientation << std::endl;
            
            // Draw orientation arrow
            cv::Point center(grasp.bbox.x + grasp.bbox.width/2, 
                            grasp.bbox.y + grasp.bbox.height/2);
            cv::Point direction(50 * cos(grasp.object_angle), 
                              50 * sin(grasp.object_angle)); // x-axis from left to right
            // cv::arrowedLine(result, center, center + direction, 
            //               cv::Scalar(0, 255, 0), 2);
        }

        // Show results
        cv::imshow("Result", result);
        cv::waitKey(0);
        pipe.stop();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}