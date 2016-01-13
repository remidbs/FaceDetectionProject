#ifndef FACEDET_ADABOOST_H_H
#define FACEDET_ADABOOST_H_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

#include "features.h"


class AdaBoost {
public:
    AdaBoost();

    void train(const int steps = 10, bool verbose = false);

    void print();

private:

    std::vector<cv::Mat> positiveSet;
    std::vector<cv::Mat> negativeSet;
    unsigned long datasetSize;
    unsigned long positiveDatasetSize;
    unsigned long negativeDatasetSize;
    std::vector<feature> features;
    std::vector<std::vector<double> > coef;
    std::vector<double> sums;
    std::vector<std::pair<int, double> > classifiers;
    std::vector<double> betas;
    std::vector<std::vector<double> > featureValues;
    std::ofstream outputfile;

    double errorWeakClassifier(double threshold, int polarisation, int step, int featureNumber);

    int weakClassifier(double threshold, int polarisation, double x);
};

void printClassifier(double threshold, int polarisation);


#endif //FACEDET_ADABOOST_H_H
