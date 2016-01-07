#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "image_tools.h"

using namespace std;
using namespace cv;

Mat normalizeImage(const Mat &A) {
    int width = A.cols;
    int height = A.rows;
    int sum = 0;
    int squareSum = 0;
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            int a = (int) A.at<uchar>(j, i);
            sum += a;
            squareSum += a * a;
        }
    double avg = (double) sum / (double) (height * width);
    double squareAvg = (double) squareSum / (double) (height * width);
    double sigma = sqrt(squareAvg - avg * avg);
    Mat B(width, height, CV_32F);
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
            B.at<float>(i, j) = (((double) A.at<uchar>(i, j) - avg) / (width * height * sigma));
    return B;

}

void integralImage(Mat &original, Mat &II) {

    int width = original.cols;
    int height = original.rows;

    Mat S(width, height, CV_32F);

    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {

            if (y == 0)
                S.at<float>(y, x) = 0;
            else
                S.at<float>(y, x) = S.at<float>(y - 1, x) + (float) (original.at<float>(y, x));

            if (x == 0)
                II.at<float>(y, x) = 0;
            else
                II.at<float>(y, x) = II.at<float>(y, x - 1) + S.at<float>(y, x);
        }
    }
}

Mat getIntegralImageFromFilename(string filename) {
    Mat I = imread("../pics/output/" + filename + ".jpeg");
    Mat I2;
    cvtColor(I, I2, CV_BGR2GRAY);
    Mat I3(I2.cols, I2.rows, CV_32F);
    I3 = normalizeImage(I2);
    Mat I4(I2.cols, I2.rows, CV_32F);
    integralImage(I3, I4);
    return I4;
}