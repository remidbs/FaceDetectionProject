#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "features.h"

using namespace cv;
using namespace std;


feature::feature() {
    type = 0;
    x = 0;
    y = 0;
    w = 0;
    h = 0;
}

feature::feature(int type, int x, int y, int w, int h) {
    assert(type < 5);
    this->type = type;
    this->x = x;
    this->y = y;
    this->w = w;
    this->h = h;
}

void feature::print() {
    switch (type) {
        case 0:
            cout << "Edge Vertical : ";
            break;
        case 1:
            cout << "Edge Horizontal : ";
            break;
        case 2:
            cout << "Line Vertical : ";
            break;
        case 3:
            cout << "Line Horizontal : ";
            break;
        default:
            cout << "Block : ";
    }
    cout << "x=" << x;
    cout << ", y=" << y;
    cout << ", w=" << w;
    cout << ", h=" << h;
    cout << endl;
}

double feature::edgeV(int x, int y, int w, int h, const Mat &II) {
    assert(w % 2 == 0);
    assert((x + w < II.cols) && (y + h < II.rows));


    //    a--b--c
    //    |  |  |
    //    d--e--f

    double a = II.at<float>(y, x);
    double b = II.at<float>(y, x + w / 2);
    double c = II.at<float>(y, x + w);
    double d = II.at<float>(y + h, x);
    double e = II.at<float>(y + h, x + w / 2);
    double f = II.at<float>(y + h, x + w);

    return (a - 2 * b + c - d + 2 * e - f);
}

double feature::edgeH(int x, int y, int w, int h, const Mat &II) {
    assert(h % 2 == 0);
    assert((x + w < II.cols) && (y + h < II.rows));

    //    a----b
    //    |    |
    //    c----d
    //    |    |
    //    e----f

    double a = II.at<float>(y, x);
    double b = II.at<float>(y, x + w);
    double c = II.at<float>(y + h / 2, x);
    double d = II.at<float>(y + h / 2, x + w);
    double e = II.at<float>(y + h, x);
    double f = II.at<float>(y + h, x + w);

    return (a - b - 2 * c + 2 * d + e - f);
}

double feature::lineV(int x, int y, int w, int h, const Mat &II) {
    assert(w % 3 == 0);
    assert((x + w < II.cols) && (y + h < II.rows));

    //    a--b--c--d
    //    |  |  |  |
    //    e--f--g--h

    double a = II.at<float>(y, x);
    double b = II.at<float>(y, x + w / 3);
    double c = II.at<float>(y, x + (2 * w) / 3);
    double d = II.at<float>(y + h, x + w);
    double e = II.at<float>(y + h, x);
    double f = II.at<float>(y + h, x + w / 3);
    double g = II.at<float>(y + h, x + (2 * w) / 3);
    double H = II.at<float>(y + h, x + w);

    return (a - 2 * b + 2 * c - d - e + 2 * f - 2 * g + H);
}

double feature::lineH(int x, int y, int w, int h, const Mat &II) {
    assert(h % 3 == 0);
    assert((x + w < II.cols) && (y + h < II.rows));

    //    a----b
    //    |    |
    //    c----d
    //    |    |
    //    e----f
    //    |    |
    //    g----h

    double a = II.at<float>(y, x);
    double b = II.at<float>(y, x + w);
    double c = II.at<float>(y + h / 3, x);
    double d = II.at<float>(y + h / 3, x + w);
    double e = II.at<float>(y + (2 * h) / 3, x);
    double f = II.at<float>(y + (2 * h) / 3, x + w);
    double g = II.at<float>(y + h, x);
    double H = II.at<float>(y + h, x + w);

    return (a - b - 2 * c + 2 * d + 2 * e - 2 * f - g + H);
}

double feature::block(int x, int y, int w, int h, const Mat &II) {
    assert(w % 2 == 0);
    assert(h % 2 == 0);
    assert((x + w < II.cols) && (y + h < II.rows));


    //    a--b--c
    //    |  |  |
    //    d--e--f
    //    |  |  |
    //    g--h--i

    double a = II.at<float>(y, x);
    double b = II.at<float>(y, x + w / 2);
    double c = II.at<float>(y, x + w);
    double d = II.at<float>(y + h / 2, x);
    double e = II.at<float>(y + h / 2, x + w / 2);
    double f = II.at<float>(y + h / 2, x + w);
    double g = II.at<float>(y + h, x);
    double H = II.at<float>(y + h, x + w / 2);
    double i = II.at<float>(y + h, x + w);

    return (a - 2 * b + c - 2 * d + 4 * e - 2 * f + g - 2 * H + i);
}

const double feature::eval(const Mat &II) {
    int width = II.cols;
    int height = II.rows;
    int xx = (width * x) / 24;
    int yy = (height * y) / 24;
    int ww = (width * w) / 24;
    int hh = (height * h) / 24;
    switch (type) {
        case 0:
            return (edgeV(xx, yy, ww - (ww % 2), hh, II));
        case 1:
            return (edgeH(xx, yy, ww, hh - (hh % 2), II));
        case 2:
            return (lineV(xx, yy, ww - (ww % 3), hh, II));
        case 3:
            return (lineH(xx, yy, ww, hh - (hh % 3), II));
        default:
            return (block(xx, yy, ww - (ww % 2), hh - (hh % 2), II));
    }
}

vector<feature> featuresIndex(int width, int height) {
    vector<feature> index;
    //edgeVertical
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int w = 2; w < width - x; w = w + 2) {
                for (int h = 1; h < height - y; h++) {
                    index.push_back(feature(0, x, y, w, h));
                }
            }
        }
    }

    //edgeHorizontal
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int w = 1; w < width - x; w++) {
                for (int h = 2; h < height - y; h = h + 2) {
                    index.push_back(feature(1, x, y, w, h));
                }
            }
        }
    }
    //lineVertical
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int w = 3; w < width - x; w = w + 3) {
                for (int h = 1; h < height - y; h++) {
                    index.push_back(feature(2, x, y, w, h));
                }
            }
        }
    }
    //lineHorizontal
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int w = 1; w < width - x; w++) {
                for (int h = 3; h < height - y; h = h + 3) {
                    index.push_back(feature(3, x, y, w, h));
                }
            }
        }
    }
    //block
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int w = 2; w < width - x; w = w + 2) {
                for (int h = 2; h < height - y; h = h + 2) {
                    index.push_back(feature(4, x, y, w, h));
                }
            }
        }
    }
    return index;
}

vector<double> evaluateFeatures(vector<feature> &features, const Mat &II) {
    vector<double> res;
    for (int i = 0; i < features.size(); i++) {
        res.push_back(features[i].eval(II));
    }
    return res;
}

