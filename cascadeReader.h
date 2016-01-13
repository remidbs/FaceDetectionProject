#ifndef FACEDET_CASCADEREADER_H
#define FACEDET_CASCADEREADER_H

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

//revoie l'étape jusqu'à laquelle est allé le détecteur basé sur la cascade. 24 est la dernière étape, et correspond
//donc à la valeur renvoyé si l'image est un visage
int detectFace(std::string filename, bool verbose = false, bool displayFeatures = false);

int detectFace(cv::Mat I, bool verbose = false, bool displayFeatures = false);


//Applique pour différentes taille et position de fenêtres la fonction detectFace précédente
int detectBestFace(std::string filename);


//Fonction de détection de visage basée sur openCV
cv::Mat opencvDetect(std::string filename);

#endif //FACEDET_CASCADEREADER_H
