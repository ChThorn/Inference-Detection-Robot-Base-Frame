#include "TransformMatrixCamToRobot.h"
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include <fstream>
#include <Eigen/LU>
TransformMatrixCamToRobot::TransformMatrixCamToRobot(
    const std::string& intrinsic_file,
    const std::string& extrinsic_file,
    int depth_window_size) 
    : depth_window_size_(depth_window_size), is_extrinsic_valid_(false) {

    // Check if files exist
    std::ifstream intrinsic_check(intrinsic_file);
    if (!intrinsic_check.good()) {
        throw std::runtime_error("Intrinsic file not found: " + intrinsic_file);
    }
    std::ifstream extrinsic_check(extrinsic_file);
    if (!extrinsic_check.good()) {
        throw std::runtime_error("Extrinsic file not found: " + extrinsic_file);
    }

    // Load camera parameters
    if (!loadMatrixFromYAML(intrinsic_file, "camera_matrix", params_.camera_matrix) ||
        !loadMatrixFromYAML(intrinsic_file, "distortion_coefficients", params_.dist_coeffs)) {
        throw std::runtime_error("Failed to load intrinsic parameters");
    }

    // Load and validate extrinsic matrix
    cv::Mat extrinsic_cv;
    if (!loadMatrixFromYAML(extrinsic_file, "extrinsic_matrix", extrinsic_cv) ||
        extrinsic_cv.rows != 4 || extrinsic_cv.cols != 4) {
        throw std::runtime_error("Invalid extrinsic matrix dimensions");
    }
    cv::cv2eigen(extrinsic_cv, params_.extrinsic);
    validateExtrinsic(params_.extrinsic); // Call non-const method
}

void TransformMatrixCamToRobot::validateExtrinsic(const Eigen::Matrix4d& extrinsic) {
    Eigen::Matrix3d rot = extrinsic.block<3,3>(0,0);
    double det = rot.determinant();
    if (std::abs(det - 1.0) > 1e-3) {
        throw std::runtime_error("Extrinsic rotation matrix is not orthonormal (det=" + 
                                std::to_string(det) + ")");
    }
    is_extrinsic_valid_ = true; // Now allowed to modify member variable
}

bool TransformMatrixCamToRobot::loadMatrixFromYAML(
    const std::string& filename,
    const std::string& key,
    cv::Mat& mat) const {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs[key] >> mat;
    return !mat.empty();
}

Eigen::Vector3d TransformMatrixCamToRobot::pixelToCamera(
    const cv::Point& pixel, float depth) const {
    std::vector<cv::Point2f> distorted{pixel};
    std::vector<cv::Point2f> undistorted;
    
    cv::undistortPoints(distorted, undistorted, 
                       params_.camera_matrix, 
                       params_.dist_coeffs,
                       cv::noArray(), params_.camera_matrix); // Ensure correct projection
    
    return Eigen::Vector3d(
        undistorted[0].x * depth,
        undistorted[0].y * depth,
        depth
    );
}

Eigen::Vector3d TransformMatrixCamToRobot::transformToRobot(
    const Eigen::Vector3d& point_cam) const {
    if (!is_extrinsic_valid_) {
        throw std::runtime_error("Extrinsic matrix is invalid");
    }
    const Eigen::Vector4d cam_homog(point_cam.x(), point_cam.y(), point_cam.z(), 1.0);
    const Eigen::Vector4d robot_homog = params_.extrinsic * cam_homog;
    return robot_homog.head<3>() / robot_homog.w();
}

float TransformMatrixCamToRobot::calculateDepth(
    const cv::Mat& depth, const cv::Point& point) const {
    std::vector<float> valid_depths;
    const int window = depth_window_size_;
    
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
    
    if (valid_depths.empty()) return std::numeric_limits<float>::quiet_NaN();
    std::nth_element(valid_depths.begin(), 
                    valid_depths.begin() + valid_depths.size()/2,
                    valid_depths.end());
    return valid_depths[valid_depths.size()/2];
}