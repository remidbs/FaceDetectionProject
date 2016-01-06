#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;


int detectFace(string filename, bool verbose = false, bool displayNormalizedImage = false,
               bool displayFeatures = false, bool displayOCVResult = false);

int detectFace(Mat I, bool verbose = false, bool displayNormalizedImage = false,
               bool displayFeatures = false, bool displayOCVResult = false);

int detectBestFace(string filename);

void opencvDetect(string filename);

int detectBestFace(string filename) {
    string path = "../pics/" + filename + ".jpeg";
    Mat I = imread(path);
    Mat I2;
    cvtColor(I, I2, CV_BGR2GRAY);
    int bestStageReached = -1;
    Rect bestWindow;
    int minSideSize = min(I2.cols, I2.rows);
    for (int i = 38; i < minSideSize; i += 1) {
        cout << i << endl;
        for (int shiftX = 0; shiftX < I2.cols - i; shiftX += 1) {
            for (int shiftY = 0; shiftY < I2.rows - i; shiftY += 1) {
                Mat underI(i, i, I2.type());
                for (int j = 0; j < i; j++) {
                    for (int k = 0; k < i; k++) {
                        underI.at<uchar>(k, j) = I2.at<uchar>(k + shiftY, j + shiftX);
                    }
                }
                int det = detectFace(underI);
                cout << det << " " << Rect(shiftX, shiftY, i, i) << endl;
                if (det >= bestStageReached) {
                    bestStageReached = det;
                    bestWindow = Rect(shiftX, shiftY, i, i);
//                    Mat I3;
//                    cvtColor(I, I3, CV_BGR2GRAY);
//                    rectangle(I3, bestWindow, Scalar(255, 0, 0), 4, 8, 0);
//                    imshow(" ", I3);
//            waitKey();
                }
            }
//            cout << "\t" << bestStageReached << " " << bestWindow << endl;
        }
    }
    cout << "Best window reached stage " << bestStageReached << endl;
//    waitKey();
    return bestStageReached;
}

int detectFace(string filename, bool verbose, bool displayNormalizedImage,
               bool displayFeatures, bool displayOCVResult) {
    string path = "../pics/" + filename + ".jpeg";
    Mat I = imread(path);
    Mat A;
    cvtColor(I, A, CV_BGR2GRAY);
    return detectFace(A, verbose, displayNormalizedImage, displayFeatures, displayOCVResult);
}

int detectFace(Mat A, bool verbose, bool displayNormalizedImage,
               bool displayFeatures, bool displayOCVResult) {
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
    if (verbose)
        cout << avg << "\t" << sigma << endl;
    Mat B(width, height, CV_32F);
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            B.at<float>(i, j) = (((double) A.at<uchar>(i, j) - avg) / (width*height*sigma));
        }
    if (displayNormalizedImage) {
        imshow("I", B);
        waitKey();
    }
    string filename2 = "../haarcascade_frontalface_default.xml";
    FileStorage fs;
    fs.open(filename2, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename2 << endl;
        return -1;
    }
    FileNode n = fs["haarcascade_frontalface_default"];
    FileNode s = n["size"];
    FileNode stages = n["stages"];
    FileNodeIterator stagesIterator = stages.begin(), stagesItEnd = stages.end();
    int numStage = -1;
    bool isAFace = true;
    for (; stagesIterator != stagesItEnd; ++stagesIterator) {
        numStage++;
        if (verbose)
            cout << endl << endl << "Beginning of stage " << numStage << "..." << endl;
        FileNode trees = (*stagesIterator)["trees"];
        double stageThreshold = (*stagesIterator)["stage_threshold"];
        int numTree = -1;
        double stageSum = 0;
        for (FileNodeIterator treeIt = trees.begin(); treeIt != trees.end(); ++treeIt) {
            numTree++;
            if (verbose)
                cout << "\t" << "Tree " << numTree << " " << endl;

            FileNode currentTree = (*((*treeIt).begin()));
            double threshold = (double) (currentTree["threshold"]);
            double leftVal = (double) (currentTree["left_val"]);
            double rightVal = (double) (currentTree["right_val"]);
            FileNode rects = currentTree["feature"]["rects"];
            int numRects = -1;
            double haarSum = 0;
            Mat layer;
            if (displayFeatures) {
                layer = Mat(A.cols, A.rows, CV_8U, (uchar) 128);
                for (int i = 0; i < A.rows; i++) {
                    for (int j = 0; j < A.cols; j++) {
                        layer.at<uchar>(i, j) = A.at<uchar>(i, j);
                    }
                }
            }
            for (FileNodeIterator rectsIt = rects.begin(); rectsIt != rects.end(); ++rectsIt) {
                numRects++;
                if (verbose)
                    cout << "\t\tRect " << numRects << " : ";

                FileNodeIterator currentRect = ((*rectsIt).begin());
                int x = (width * (int) *currentRect) / 24;
                ++currentRect;
                int y = (width * (int) *currentRect) / 24;
                ++currentRect;
                int dx = (width * (int) *currentRect) / 24;
                ++currentRect;
                int dy = (width * (int) *currentRect) / 24;
                ++currentRect;
                double coef = (double) *currentRect;
                if (x + dx > width)
                    cerr << x + dx << endl;
                if (y + dy > height)
                    cerr << y + dy << endl;
                if (verbose)
                    cout << x << " " << y << " " << dx << " " << dy << " " << coef << endl;

                for (int i = 0; i < dx; i++) {
                    for (int j = 0; j < dy; j++) {
                        haarSum += coef * ((double) B.at<float>(y + j, x + i));
                        if (displayFeatures)
                            layer.at<uchar>(y + j, x + i) = layer.at<uchar>(y + j, x + i) + 40 * coef;
                    }
                }
            }
            if (displayFeatures) {
                imshow("I", layer);
                waitKey();
            }
            stageSum += haarSum <= threshold ? leftVal : rightVal;
            if (verbose)
                cout << "\t\t\tfeatureValue = " << haarSum << "; threshold : " << threshold << endl;
        }
        if(verbose)
            cout << "stageSum = " << stageSum << "; stageThreshold = " << stageThreshold << endl;
        if (stageSum < stageThreshold) {
            isAFace = false;
            break;
        }
    }

    if(verbose) {
        cout << "My version : ";
        if (isAFace)
            cout << "this is a face!" << endl;
        else
            cout << "this is not a face. Discarded at stage " << numStage << endl;
    }
    return numStage;
}

void opencvDetect(string filename) {
    string path = "../pics/" + filename + ".jpeg";
    string face_cascade_name = "../haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    string window_name = "Capture - Face detection";
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
        return;
    };

    Mat frame = imread(path);
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    std::vector<Rect> faces;

    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    cout << "OpenCV version : ";
    if (faces.size() > 0)
        cout << "there is at least one face on the picture..." << endl;
    else
        cout << "no face on this picture" << endl;
    for (size_t i = 0; i < faces.size(); i++) {
        cout << faces[i];
        Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8,
                0);
        rectangle(frame, faces[i], Scalar(255, 0, 0), 4, 8, 0);
    }
    //-- Show what you got
    imshow(window_name, frame);
    waitKey();

}