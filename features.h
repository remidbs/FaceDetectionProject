#ifndef FACEDET_FEATURES_H
#define FACEDET_FEATURES_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


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

    static double edgeV(int x, int y, int w, int h, const cv::Mat &II);

    static double edgeH(int x, int y, int w, int h, const cv::Mat &II);

    static double lineV(int x, int y, int w, int h, const cv::Mat &II);

    static double lineH(int x, int y, int w, int h, const cv::Mat &II);

    static double block(int x, int y, int w, int h, const cv::Mat &II);

    const double eval(const cv::Mat &II);

    std::string getString();
};

std::vector<feature> featuresIndex(int width = 24, int height = 24);

std::vector<double> evaluateFeatures(std::vector<feature> &features, const cv::Mat &II);

#endif //FACEDET_FEATURES_H
