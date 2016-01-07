#ifndef FACEDET_ADABOOST_H_H
#define FACEDET_ADABOOST_H_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "features.h"
#include "image_tools.h"


class AdaBoost {
public:
    AdaBoost(std::vector<std::string> positiveFilenames, std::vector<std::string> negativeFilenames);

    void train(const int steps = 10, bool verbose = false);

    void print();

private:

    std::vector<cv::Mat> positiveSet;
    std::vector<cv::Mat> negativeSet;
    unsigned long datasetSize;
    std::vector<feature> features;
    std::vector<std::vector<double> > coef;
    std::vector<double> sums;
    std::vector<std::pair<int, double> > classifiers;
    std::vector<double> betas;

    double errorWeakClassifier(double threshold, int polarisation, int step, int featureNumber);

    int weakClassifier(double threshold, int polarisation, double x);
};


#endif //FACEDET_ADABOOST_H_H
