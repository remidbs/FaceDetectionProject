#ifndef FACEDET_CASCADEREADER_H
#define FACEDET_CASCADEREADER_H

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

int detectFace(std::string filename, bool verbose = false, bool displayFeatures = false);

int detectFace(cv::Mat I, bool verbose = false, bool displayFeatures = false);

int detectBestFace(std::string filename);

void opencvDetect(std::string filename);

int detectBestFace(std::string filename);

#endif //FACEDET_CASCADEREADER_H
