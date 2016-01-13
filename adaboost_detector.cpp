#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include "adaboost_detector.h"
#include "image_tools.h"
#include "features.h"

using namespace std;
using namespace cv;

bool AdaboostDetector::detectFace(std::string filename) {
    Mat I = getIntegralImageFromFilename(filename);
    vector<double> featureValues = evaluateFeatures(features, I);
    double detectorVal = 0;
    for (int i = 0; i < featureValues.size(); i++) {
        detectorVal += coefficients[i] * classifiers[i].getLabel(featureValues[i]);
    }
    return detectorVal <= poolThreshold;
}

AdaboostDetector::AdaboostDetector() {
    ifstream detector;
    detector.open("../output");
    poolThreshold = 0;
    for (int i = 0; i < 15; i++) {
        double coef, threshold;
        int featureType, x, y, w, h, polarisation;
        detector >> coef >> featureType >> x >> y >> w >> h >> polarisation >> threshold;
        coef = log(1-coef)-log(coef);
        coefficients.push_back(coef);
        features.push_back(feature(featureType, x, y, w, h));
        classifiers.push_back(Classifier(polarisation, threshold));
        poolThreshold += coef / 2;
    }

}

Classifier::Classifier(int pol, double thresh) {
    polarisation = pol;
    threshold = thresh;
}

int Classifier::getLabel(double val) {
    if (polarisation * val < threshold * polarisation)
        return 0;
    else
        return 1;
}
