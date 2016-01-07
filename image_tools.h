#ifndef FACEDET_IMAGE_TOOLS_H
#define FACEDET_IMAGE_TOOLS_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat normalizeImage(const cv::Mat &A);

void integralImage(cv::Mat &original, cv::Mat &II);

cv::Mat getIntegralImageFromFilename(std::string filename);

#endif //FACEDET_IMAGE_TOOLS_H
