#ifndef FACEDET_FEATURES_H
#define FACEDET_FEATURES_H

#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>


class feature {
public :
    int x;
    int y;
    int w;
    int h;
    int type;

    feature();

    feature(int type, int x, int y, int w, int h);

    void print();

    static double edgeV(int x, int y, int w, int h, cv::Mat &II);

    static double edgeH(int x, int y, int w, int h, cv::Mat &II);

    static double lineV(int x, int y, int w, int h, cv::Mat &II);

    static double lineH(int x, int y, int w, int h, cv::Mat &II);

    static double block(int x, int y, int w, int h, cv::Mat &II);

    double eval(cv::Mat &II);

};

std::vector<feature> featuresIndex(int width = 24, int height = 24);

#endif //FACEDET_FEATURES_H
