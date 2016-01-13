#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "cascadeReader.h"
#include "image_tools.h"

using namespace std;
using namespace cv;

int detectBestFace(string filename) {
    string path = "../pics/" + filename + ".jpeg";
    Mat I = imread(path);
    Mat I2;
    cvtColor(I, I2, CV_BGR2GRAY);
    int bestStageReached = -1;
    Rect bestWindow;
    int minSideSize = min(I2.cols, I2.rows);
    for (int i = 80; i < minSideSize; i += 4) {
        cout << "Searching for face of size " << i << endl;
        for (int shiftX = 0; shiftX < I2.cols - i; shiftX += 4) {
            for (int shiftY = 0; shiftY < I2.rows - i; shiftY += 4) {
                Rect window(shiftX, shiftY, i, i);
//                Mat underI(i, i, CV_8U);
//                I2(window).copyTo(underI);
                Mat underI = I2(window);
                int det = detectFace(underI);
                if (det >= bestStageReached) {
                    bestStageReached = det;
                    bestWindow = window;
                    if (bestStageReached == 24)
                        break;
                }
            }
        }
    }
    rectangle(I2, bestWindow, Scalar(255, 0, 0), 4, 8, 0);
    imshow("Best face found", I2);
    waitKey();
    return bestStageReached;
}

int detectFace(string filename, bool verbose,
               bool displayFeatures) {
    Mat I = imread(filename);
    Mat A;
    cvtColor(I, A, CV_BGR2GRAY);
    return detectFace(A, verbose, displayFeatures);
}

int detectFace(Mat A, bool verbose,
               bool displayFeatures) {
    int width = A.cols;
    int height = A.rows;

    Mat B(width, height, CV_32F);
    B = normalizeImage(A);

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
        if (verbose)
            cout << "stageSum = " << stageSum << "; stageThreshold = " << stageThreshold << endl;
        if (stageSum < stageThreshold) {
            isAFace = false;
            break;
        }
    }

    if (verbose) {
        if (isAFace)
            cout << "This is a face!" << endl;
        else
            cout << "This is not a face. Discarded at stage " << numStage << endl;
    }
    return numStage;
}

Mat opencvDetect(string filename) {
    string path = filename;
    string face_cascade_name = "../haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    string window_name = "Capture - Face detection";
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
        throw std::logic_error("File not found");
    };

    Mat frame = imread(path);
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    std::vector<Rect> faces;

    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    cout << "OpenCV version : ";
    if (faces.size() > 0) {
        Mat Res;
        frame(faces[0]).copyTo(Res);
        cout << "there is at least one face on the picture..." << endl;
        return Res;
    } else
        cout << "no face on this picture" << endl;
    for (size_t i = 0; i < faces.size(); i++) {
        cout << faces[i];
        Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        rectangle(frame, faces[i], Scalar(255, 0, 0), 4, 8, 0);
    }

    throw std::logic_error("No face !");


    imshow(window_name, frame);
    waitKey();

}