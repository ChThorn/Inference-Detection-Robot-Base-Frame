#ifndef TRANSFORM_MATRIX_CAM_TO_ROBOT_H
#define TRANSFORM_MATRIX_CAM_TO_ROBOT_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <string>
#include <stdexcept>

struct CameraParams {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    Eigen::Matrix4d extrinsic;
};

class TransformMatrixCamToRobot {
public:
    TransformMatrixCamToRobot(const std::string& intrinsic_file, 
                             const std::string& extrinsic_file,
                             int depth_window_size = 2);

    Eigen::Vector3d pixelToCamera(const cv::Point& pixel, float depth) const;    
    Eigen::Vector3d transformToRobot(const Eigen::Vector3d& point_cam) const;
    float calculateDepth(const cv::Mat& depth, const cv::Point& point) const;

    const CameraParams& getCameraParams() const { return params_; }
    bool isExtrinsicValid() const { return is_extrinsic_valid_; }

private:
    CameraParams params_;
    bool is_extrinsic_valid_;
    int depth_window_size_;

    void validateExtrinsic(const Eigen::Matrix4d& extrinsic); // Removed "const" qualifier
    bool loadMatrixFromYAML(const std::string& filename, 
                           const std::string& key, 
                           cv::Mat& mat) const;
};

#endif