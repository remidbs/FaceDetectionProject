#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#include "features.h"
#include "image_tools.h"
#include "adaboost.h"
#include "cascadeReader.h"
#include "adaboost_detector.h"

int main() {

    //--To run the training step, uncomment next lines---
//    AdaBoost adaBoost;
//    adaBoost.train(15, true);
//    adaBoost.print();

    //--To run detector step for the previously trained classifier uncomment next lines---
    AdaboostDetector detector;
    int truePositive = 0;
    int trueNegative = 0;
    int falsePositive = 0;
    int falseNegative = 0;
    for (int i = 0; i < 9; i++) {
        if (detector.detectFace("../pics/validation_set/positive" + to_string(i) + ".jpeg"))
            truePositive++;
        else
            falseNegative++;
    }
    for (int i = 0; i < 6; i++) {
        if (detector.detectFace("../pics/validation_set/negative" + to_string(i) + ".jpeg"))
            falsePositive++;
        else
            trueNegative++;
    }
    cout << "Confusion matrix" << endl;
    cout << "\tT\tF" << endl;
    cout << "P\t" << truePositive << "\t" << falsePositive << endl;
    cout << "N\t" << trueNegative << "\t" << falseNegative << endl;


    //--To run the face detector, uncomment next lines---

//    int res = detectFace("../pics/man224.jpeg");
//    if(res == 24){
//        cout << "This picture is a face" << endl;
//    } else{
//        cout << "This picture is not a face" << endl;
//    }

    //--To run the algorithm finding a window containing a face in an image, uncomment next lines---

//    detectBestFace("baseface_woman_small");

    //--To run the evaluation of the classifiers contained in the xml file, uncomment next lines---

    truePositive = 0;
    trueNegative = 0;
    falsePositive = 0;
    falseNegative = 0;
    for (int i = 1; i < 115; i++) {
        if (detectFace("../pics/training_set/positive" + to_string(i) + ".jpeg") == 24 )
            truePositive++;
        else
            falseNegative++;
    }
    for (int i = 0; i < 14; i++) {
        if (detectFace("../pics/training_set/negative" + to_string(i) + ".jpeg") == 24)
            falsePositive++;
        else
            trueNegative++;
    }
    cout << "Confusion matrix" << endl;
    cout << "\tT\tF" << endl;
    cout << "P\t" << truePositive << "\t" << falsePositive << endl;
    cout << "N\t" << trueNegative << "\t" << falseNegative << endl;

    return 0;
}
