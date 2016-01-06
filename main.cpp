#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "cascadeReader.cpp"
#include "features.cpp"

using namespace cv;
using namespace std;

int main() {
    vector<pair<string, int> > res;
    string filename;
    filename = "man240";
    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
    filename = "man48";
    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "man24";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
    filename = "manVariousSize";
    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
    filename = "woman48";
    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "random_24";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
    filename = "random48";
    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "woman24_1";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "woman24_2";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    filename = "woman24_3";
//    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
    filename = "man224";
    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
    filename = "man48upsidedown";
    res.push_back(pair<string, int>(filename, detectFace(filename, false)));
//    detectFace("manVariousSize", true, true, true);
    for (int i = 0; i < res.size(); i++)
        cout << res[i].first << " : " << res[i].second << endl;


//    int max = -1;
//    for (int i = 1500; i < 4000; i++) {
//        int res = detectFace("man48", false, false, false, false, i);
//        if (res > max) {
////            cout << endl << "Max dépassé pour i = " << i << ". Il vaut : " << res << endl;
//            max = res;
//        }
////        cout << i << " : " << res << "\t";
//        if (res == 24)
//            cout << endl << i << endl;
//    }

//      detectFace("random48", true, false, false, false, 0);
//      detectFace("man48upsidedown", true, false, false, false, 0);


//    opencvDetect("man48");
//    detectFace("man240",true, false,true);

//    string path = "../pics/man24.jpeg";
//    Mat I = imread(path);
//    Mat I2;
//    cvtColor(I, I2, CV_BGR2GRAY);
//    Mat I3(I2.cols,I2.rows,CV_32F);
//    integralImage(I2,I3);
//    map<int, feature> M = featuresIndex(I3.cols,I3.rows);
//    cout << M.size() << endl;
//    for(int i=0;i<M.size();i++){
//        M[i].print();
//    }

    return 0;
}
