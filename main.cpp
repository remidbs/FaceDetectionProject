#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "image_tools.cpp"
#include "cascadeReader.cpp"
#include "features.cpp"

using namespace cv;
using namespace std;

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
//    Mat I3(I2.cols,I2.rows,CV_32F);
//    I3 = normalizeImage(I2);
//    Mat I4(I2.cols,I2.rows,CV_32F);
//    integralImage(I3,I4);
//    map<int, feature> M = featuresIndex(I4.cols,I4.rows);
//    cout << M.size() << endl;
//    for(int i=0;i<M.size();i++){
//        cout << M[i].eval(I4) << endl;
//    }

    return 0;
}
