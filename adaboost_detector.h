#ifndef FACEDET_ADABOOST_DETECTOR_H
#define FACEDET_ADABOOST_DETECTOR_H

#include <string>
#include "features.h"

class Classifier{
private:
    int polarisation;
    double threshold;

public:
    Classifier(int pol, double thresh);
    int getLabel(double val);
};

class AdaboostDetector {
public:
    bool detectFace(std::string filename);
    AdaboostDetector();

private:
    std::vector<Classifier> classifiers;
    std::vector<feature> features;
    std::vector<double> coefficients;
    double poolThreshold;
};

#endif //FACEDET_ADABOOST_DETECTOR_H
