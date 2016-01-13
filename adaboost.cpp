#include <iostream>
#include <fstream>

#include "adaboost.h"
#include "image_tools.h"


using namespace std;
using namespace cv;

AdaBoost::AdaBoost() {
    outputfile.open("../output");
    vector<string> positiveFilenames;
    vector<string> negativeFilenames;
    for (int i = 1; i <= 106; i++) {
        positiveFilenames.push_back("../pics/training_set/positive" + to_string(i) + ".jpeg");
    }
    for (int i = 0; i < 8; i++) {
        negativeFilenames.push_back("../pics/training_set/negative" + to_string(i) + ".jpeg");
    }
    features = featuresIndex(24, 24);
    datasetSize = positiveFilenames.size() + negativeFilenames.size();
    positiveDatasetSize = positiveFilenames.size();
    negativeDatasetSize = negativeFilenames.size();
    coef.push_back(vector<double>());
    for (int i = 0; i < positiveDatasetSize; i++) {
        coef[0].push_back(1. / positiveDatasetSize / 2.);
        Mat I = getIntegralImageFromFilename(positiveFilenames[i]);
        positiveSet.push_back(I);
        featureValues.push_back(evaluateFeatures(features, positiveSet[i]));
    }
    for (int i = 0; i < negativeDatasetSize; i++) {
        coef[0].push_back(1. / negativeDatasetSize / 2.);
        negativeSet.push_back(getIntegralImageFromFilename(negativeFilenames[i]));
        featureValues.push_back(evaluateFeatures(features, negativeSet[i]));
    }
    sums.push_back(1);
}

void AdaBoost::train(const int steps, bool verbose) {
    for (int step = 1; step <= steps; step++) {
        if (verbose)
            cout << "Step " << step << endl;
        sums[step] = 0;
        cout << "---Beginning normalization of coefs..." << endl;
        for (int i = 0; i < datasetSize; i++) {
            coef[step - 1][i] = coef[step - 1][i] / sums[step - 1];
        }
        cout << "---end" << endl << endl;

        cout << "---Beginning of research of classifier minimizing error..." << endl;
        double errorMinAmongFeatures = numeric_limits<double>::max();
        double thresholdOptAmongFeatures = -1;
        int polarisationOptAmongFeatures = -1;
        int optFeatureNb = -1;
        for (int featureNumber = 0; featureNumber < features.size(); featureNumber++) {
            if (verbose) {
                if (featureNumber % (features.size() / 20) == 0) {
                    cout << "\t" << featureNumber * 100 / features.size() << "%" << endl;
                }
            }
            int polarisationOpt = -1;
            double thresholdOpt = -1;
            double errorMin = numeric_limits<double>::max();
            for (int polarisation = -1; polarisation < 2; polarisation += 2) {
                for (int data = 0; data <= datasetSize; data++) {
                    double threshold = 0;
                    if (data == datasetSize) {
                        threshold = featureValues[data - 1][featureNumber] + 1;
                    } else if (data == 0) {
                        threshold = featureValues[0][featureNumber] - 1;
                    } else {
                        threshold = (featureValues[data - 1][featureNumber] + featureValues[data][featureNumber]) / 2;
                    }
                    double error = errorWeakClassifier(threshold, polarisation, step, featureNumber);
                    if (error < errorMin) {
                        errorMin = error;
                        thresholdOpt = threshold;
                        polarisationOpt = polarisation;
                    }
                }
            }
            if (errorMinAmongFeatures > errorMin) {
                errorMinAmongFeatures = errorMin;
                polarisationOptAmongFeatures = polarisationOpt;
                thresholdOptAmongFeatures = thresholdOpt;
                optFeatureNb = featureNumber;
                cout << "\terrorMinAmongFeatures : " << errorMinAmongFeatures << " ";
                printClassifier(thresholdOptAmongFeatures, polarisationOptAmongFeatures);
                cout << endl;
                cout << "\t";
                features[optFeatureNb].print();
            }
        }
        cout << "100%" << endl;
        cout << "---end-- minimal error :" << errorMinAmongFeatures << endl << endl;

        outputfile << errorMinAmongFeatures << " " << features[optFeatureNb].getString() << " " <<
        polarisationOptAmongFeatures << " " <<
        thresholdOptAmongFeatures << endl;

        classifiers.push_back(pair<int, double>(polarisationOptAmongFeatures, thresholdOptAmongFeatures));
        betas.push_back(errorMinAmongFeatures / (1 - errorMinAmongFeatures));
        sums[step] = 0;
        coef.push_back(vector<double>());
        cout << "---Beginning update of coefs..." << endl;
        for (int i = 0; i < datasetSize; i++) {
            coef[step].push_back(coef[step - 1][i]);
            if ((i < positiveSet.size()
                 && weakClassifier(thresholdOptAmongFeatures, polarisationOptAmongFeatures,
                                   features[optFeatureNb].eval(positiveSet[i])) == 1) ||
                (i >= positiveSet.size()
                 && weakClassifier(thresholdOptAmongFeatures, polarisationOptAmongFeatures,
                                   features[optFeatureNb].eval(negativeSet[i - positiveSet.size()])) == 0)) {
                coef[step][i] *= betas[step - 1];
            }
            sums[step] += coef[step][i];
        }
        cout << "---end" << endl << endl;
    }
}

void AdaBoost::print() {
    for (int i = 0; i < classifiers.size(); i++) {
        cout << "Classifier " << i << " : [polarisation: " << classifiers[i].first << "; threshold: " <<
        classifiers[i].second << "]" << endl;
    }
}

double AdaBoost::errorWeakClassifier(double threshold, int polarisation, int step, int featureNumber) {
    double error = 0;
    for (int j = 0; j < datasetSize; j++) {
        if (j < positiveSet.size())
            error += coef[step - 1][j] *
                     abs(weakClassifier(threshold, polarisation, featureValues[j][featureNumber]) - 1);
        else
            error += coef[step - 1][j] * weakClassifier(threshold, polarisation, featureValues[j][featureNumber]);
    }
    return error;
}

int AdaBoost::weakClassifier(double threshold, int polarisation, double x) {
    if (polarisation * x < polarisation * threshold)
        return 1;
    else
        return 0;
}

void printClassifier(double threshold, int polarisation) {
    cout << "[polarisation: " << polarisation << "; threshold: " << threshold << "]";
}