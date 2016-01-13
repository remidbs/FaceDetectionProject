#ifndef FACEDET_FEATURES_H
#define FACEDET_FEATURES_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/*Cette classe a pour but de gerer les coefficients de Haar afin de faciliter leur manipulation dans la suite.*/

class feature {
public :
    int x; //abcisse du pixel superieur gauche de la zone de calcul
    int y; //ordonnée du pixel superieur gauche de la zone de calcul
    int w; //largeur de la zone de calcul
    int h; //hauteur de la zone de calcul
    int type; //type de coefficient (0,1,2,3 ou 4)

    feature();

    feature(int type, int x, int y, int w, int h);

    void print();

    //fonctions pour évaluer le feature sur une image en particulier
    static double edgeV(int x, int y, int w, int h, const cv::Mat &II);

    static double edgeH(int x, int y, int w, int h, const cv::Mat &II);

    static double lineV(int x, int y, int w, int h, const cv::Mat &II);

    static double lineH(int x, int y, int w, int h, const cv::Mat &II);

    static double block(int x, int y, int w, int h, const cv::Mat &II);

    //appelle les fonctions ci-dessus selon de type de feature
    const double eval(const cv::Mat &II);

    std::string getString();
};

/*Cette fonction a pour but de stocker l'ensemble des features possibles sur une image 24x24, et de leur assigner un indice unique.*/

std::vector<feature> featuresIndex(int width = 24, int height = 24);

/*Cette fonction a pour but d'évaluer les features créés par la fonction featuresIndex sur une image en particulier
 * La matrice II doit être une image intégrale*/

std::vector<double> evaluateFeatures(std::vector<feature> &features, const cv::Mat &II);

#endif //FACEDET_FEATURES_H
