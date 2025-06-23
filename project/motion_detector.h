#ifndef MOTION_DETECTOR_H
#define MOTION_DETECTOR_H

#include <opencv2/opencv.hpp>

class MotionDetector {
public:
    MotionDetector();
    void processVideo(const std::string& filename);

private:
    cv::Mat rectKernel;
};

#endif

