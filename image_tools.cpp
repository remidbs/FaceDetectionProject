#include <opencv2/imgproc/imgproc.hpp>

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