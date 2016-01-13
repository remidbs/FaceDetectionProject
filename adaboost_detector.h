#ifndef FACEDET_ADABOOST_DETECTOR_H
#define FACEDET_ADABOOST_DETECTOR_H

#include <string>
#include "features.h"

//Classe représentant un classifieur
class Classifier{
private:
    int polarisation;
    double threshold;

public:
    Classifier(int pol, double thresh);
    int getLabel(double val);//renvoie le label associé à une valeur, à savoir l'image par la fonction marche d'un double
};

class AdaboostDetector {
public:
    bool detectFace(std::string filename);
    AdaboostDetector();//construit le classifieur robuste à partir des classifieurs faibles, features et coefficients situés dans le fichier écrit par la classe AdaBoost

private:
    std::vector<Classifier> classifiers;
    std::vector<feature> features;
    std::vector<double> coefficients;
    double poolThreshold;
};

#endif //FACEDET_ADABOOST_DETECTOR_H
