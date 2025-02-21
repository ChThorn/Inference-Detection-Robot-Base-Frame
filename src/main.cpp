// #include <librealsense2/rs.hpp>
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include "YoloDetector.h"

// // Define default paths
// const std::string DEFAULT_MODEL_PATH = "/home/thornch/Documents/Cpp/camera_preparation/Inference-Detection-Robot-Base-Frame/yolov3.weights";
// const std::string DEFAULT_CONFIG_PATH = "/home/thornch/Documents/Cpp/camera_preparation/Inference-Detection-Robot-Base-Frame/yolov3.cfg";
// const std::string DEFAULT_INTRINSIC_FILE = "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/intrinsic_params.yml";
// const std::string DEFAULT_EXTRINSIC_FILE = "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/extrinsic_params.yml";

// int main(int argc, char** argv) {
//     // if (argc < 5) {
//     //     std::cerr << "Usage: " << argv[0] << " <model_path> <config_path> <intrinsic_file> <extrinsic_file>\n";
//     //     return -1;
//     // }

//     // Use command line arguments if provided, otherwise use defaults
//     std::string model_path = (argc > 1) ? argv[1] : DEFAULT_MODEL_PATH;
//     std::string config_path = (argc > 2) ? argv[2] : DEFAULT_CONFIG_PATH;
//     std::string intrinsic_file = (argc > 3) ? argv[3] : DEFAULT_INTRINSIC_FILE;
//     std::string extrinsic_file = (argc > 4) ? argv[4] : DEFAULT_EXTRINSIC_FILE;

//     try {
//         rs2::pipeline pipe;
//         rs2::config cfg;
//         cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
//         cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
//         pipe.start(cfg);

//         // Align depth to color
//         rs2::align align_to_color(RS2_STREAM_COLOR);

//         // YoloDetector detector(
//         //     argv[1], argv[2], argv[3], argv[4], 
//         //     0.5, 0.4, cv::Size(416, 416), 0.6
//         // );

//         YoloDetector detector(
//             model_path, config_path, intrinsic_file, extrinsic_file,
//             0.5, 0.4, cv::Size(416, 416), 0.6
//         );

//         cv::namedWindow("Detection Results", cv::WINDOW_AUTOSIZE);
//         cv::namedWindow("Depth Map", cv::WINDOW_AUTOSIZE);

//         while (cv::waitKey(1) < 0) {
//             rs2::frameset frames = pipe.wait_for_frames();
//             frames = align_to_color.process(frames); // Align depth to color

//             auto color_frame = frames.get_color_frame();
//             auto depth_frame = frames.get_depth_frame();

//             if (!color_frame || !depth_frame) continue;

//             // Convert to OpenCV
//             cv::Mat color(cv::Size(640, 480), CV_8UC3, 
//                         (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
//             cv::Mat depth(cv::Size(640, 480), CV_16UC1, 
//                         (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
//             depth.convertTo(depth, CV_32F, 1.0 / 1000.0);

//             // Normalize depth for visualization
//             double min_val, max_val;
//             cv::minMaxLoc(depth, &min_val, &max_val);
//             cv::Mat depth_vis;
//             depth.convertTo(depth_vis, CV_8UC1, 255.0 / max_val);
//             cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);

//             // Detect and display
//             cv::Mat result = detector.detect(color, depth);
//             cv::imshow("Detection Results", result);
//             cv::imshow("Depth Map", depth_vis);
//         }

//         pipe.stop();
//         cv::destroyAllWindows();
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
        
        // Start pipeline with config
        rs2::pipeline_profile profile = pipe.start(cfg);

        // Get depth scale for converting depth values
        float depth_scale = profile.get_device()
                                 .first<rs2::depth_sensor>()
                                 .get_depth_scale();

        // Create align object to align depth to color frame
        rs2::align align_to_color(RS2_STREAM_COLOR);

        // Create named windows
        cv::namedWindow("Detection Results", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth Map", cv::WINDOW_AUTOSIZE);

        // Initialize detector with all parameters
        YoloDetector detector(
            "/home/thornch/Documents/Cpp/camera_preparation/Inference-Detection-Robot-Base-Frame/yolov3.weights",
            "/home/thornch/Documents/Cpp/camera_preparation/Inference-Detection-Robot-Base-Frame/yolov3.cfg",
            "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/intrinsic_params.yml",
            "/home/thornch/Documents/Cpp/camera_preparation/camera_extrinsic_cal/extrinsic_params.yml",
            0.5,    // confidence threshold
            0.4,    // NMS threshold
            cv::Size(416, 416),  // network input size
            0.6     // grasp point x ratio
        );

        // Wait for frames to settle
        for(int i = 0; i < 30; i++) {
            pipe.wait_for_frames();
        }

        // Capture and align frames
        auto frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);

        // Get individual frames
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

        // Convert depth to meters (multiply by scale instead of dividing by 1000)
        depth.convertTo(depth, CV_32F);
        depth = depth * depth_scale;

        // Create depth visualization
        cv::Mat depth_vis;
        cv::Mat depth_normalized;
        cv::normalize(depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(depth_normalized, depth_vis, cv::COLORMAP_JET);

        // Detect objects and get poses
        cv::Mat result = detector.detect(color, depth);

        // Process detections
        for (const auto& grasp : detector.get_detections()) {
            std::cout << "Object detected:\n"
                     << "  Camera frame position (m): "
                     << grasp.position_cam.transpose() << "\n"
                     << "  Robot frame position (m): "
                     << grasp.position_robot.transpose() << "\n"
                     << "  Confidence: " << grasp.confidence << "\n"
                     << "  Orientation matrix:\n" 
                     << grasp.orientation << std::endl;

            // For visualization, draw orientation
            cv::Point center(grasp.bbox.x + grasp.bbox.width/2, 
                            grasp.bbox.y + grasp.bbox.height/2);
            cv::Point direction(30 * cos(grasp.object_angle), 
                            30 * sin(grasp.object_angle));
            cv::arrowedLine(result, center, center + direction, 
                            cv::Scalar(0, 255, 0), 2);
        }

        // Display images and wait for key press
        cv::imshow("Detection Results", result);
        cv::imshow("Depth Map", depth_vis);
        
        std::cout << "\nPress any key to exit...\n";
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