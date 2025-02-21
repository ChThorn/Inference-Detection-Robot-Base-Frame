cmake_minimum_required(VERSION 3.10.0)  # Updated to remove deprecation warning
project(Inference-Detection-Robot-Base-Frame)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${realsense2_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set source files
set(SOURCE_FILES 
    src/main.cpp
    src/YoloDetector.cpp
    src/TransformMatrixCamToRobot.cpp
)

# Create executable
add_executable(main ${SOURCE_FILES})

# Link libraries
target_link_libraries(main 
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}  # Changed from realsense2_LIBS
    realsense2
    Eigen3::Eigen
)