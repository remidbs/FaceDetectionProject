#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#include "features.h"
#include "image_tools.h"
#include "adaboost.h"
#include "cascadeReader.h"

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
//
    vector<string> positives;
    vector<string> negatives;
    for (int i = 0; i < 244; i++) {
        negatives.push_back("negative" + to_string(i));
    }
    for (int i = 0; i < 121; i++) {
        positives.push_back("positive" + to_string(i));
    }

    AdaBoost adaBoost(positives, negatives);

    adaBoost.train(1,true);

    adaBoost.print();

//    int count = 0;
//    for (int i = 0; i < 245; i++) {
//            Mat image(80, 80, CV_8U);
//            for (int x = 0; x < 80; x++)
//                for (int y = 0; y < 80; y++)
//                    image.at<uchar>(x, y) = rand() % 256;
//            imwrite("../pics/output/negative" + to_string(count) + ".jpeg", image);
//            count++;
    return 0;
}
