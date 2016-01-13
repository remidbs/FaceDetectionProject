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
//    vector<pair<string, int> > res;
//    string filename;
//    filename = "man240";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "man48";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "manVariousSize";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "woman48";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "random48";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "man224";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "man48upsidedown";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    for (int i = 0; i < res.size(); i++)
//        cout << res[i].first << " : " << res[i].second << endl;

//    opencvDetect("man48");

//    string path = "../pics/man24.jpeg";
//    Mat I = imread(path);
//    Mat I2;
//    cvtColor(I, I2, CV_BGR2GRAY);
//    Mat I3(I2.cols, I2.rows, CV_32F);
//    I3 = normalizeImage(I2);
//    Mat I4(I2.cols, I2.rows, CV_32F);
//    integralImage(I3, I4);
//    vector<feature> M = featuresIndex(I4.cols, I4.rows);
//    cout << M.size() << endl;
//    for (int i = 0; i < M.size(); i++) {
//        cout << M[i].eval(I4) << endl;
//    }


    //--To run the training step, uncomment next lines---
//    AdaBoost adaBoost;
//
//    adaBoost.train(15, true);
//
//    adaBoost.print();

    //--To run detector step for the previously trained classifier
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
//    detectBestFace("baseface_woman_small");

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
