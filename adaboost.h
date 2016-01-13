#ifndef FACEDET_ADABOOST_H_H
#define FACEDET_ADABOOST_H_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

#include "features.h"


class AdaBoost {
public:
    AdaBoost();//initialise les champs, génère les features, calcules les images intégrales normalisées associées au images test

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
    std::vector<double> sums;//somme des coefficients utilisées à chaque étape pour renormaliser les coefficients et avoir une distribution de probabilité
    std::vector<std::pair<int, double> > classifiers;//classifieurs trouvés à chaque étape
    std::vector<double> betas;//valeurs permettant d'actualiser les coefficients
    std::vector<std::vector<double> > featureValues;//valeur des features pour chaque image
    std::ofstream outputfile;//fichier de sortie, où écrire les classifieurs

    double errorWeakClassifier(double threshold, int polarisation, int step, int featureNumber);

    int weakClassifier(double threshold, int polarisation, double x);
};

void printClassifier(double threshold, int polarisation);


#endif //FACEDET_ADABOOST_H_H
