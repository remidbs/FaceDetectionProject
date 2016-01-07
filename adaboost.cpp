#include "adaboost.h"
#include "image_tools.h"

using namespace std;
using namespace cv;

AdaBoost::AdaBoost(vector<string> positiveFilenames, vector<string> negativeFilenames) {
    features = featuresIndex(24, 24);
    //coef = vector<vector<double> >();
    datasetSize = positiveSet.size() + negativeSet.size();
    for (int i = 0; i < positiveSet.size(); i++) {
        coef[0].push_back(1. / positiveSet.size() / 2.);
        positiveSet.push_back(getIntegralImageFromFilename(positiveFilenames[i]));
    }
    for (int i = 0; i < negativeSet.size(); i++) {
        coef[0].push_back(1. / negativeSet.size() / 2.);
        negativeSet.push_back(getIntegralImageFromFilename(negativeFilenames[i]));
    }
    //sums = vector<double>();
    sums.push_back(1);
    //classifiers = vector<pair<int,double> >();
//        betas = vector<double>();
}

void AdaBoost::train(const int steps, bool verbose) {
    for (int step = 1; step < steps; step++) {
        if (verbose)
            cout << "Step " << step << endl;
        sums[step] = 0;
        for (int i = 0; i < datasetSize; i++) {
            coef[step - 1][i] = coef[step - 1][i] / sums[step - 1];
        }
        double errorMinAmongFeatures = numeric_limits<double>::max();
        double thresholdOptAmongFeatures = -1;
        int polarisationOptAmongFeatures = -1;
        int optFeatureNb = -1;
        vector<double> thresholds;
        for (int featureNumber = 0; featureNumber < features.size(); featureNumber++) {
            int polarisationOpt = -1;
            double thresholdOpt = -1;
            double errorMin = numeric_limits<double>::max();
            for (int polarisation = -1; polarisation < 2; polarisation += 2) {
                double a = -1;
                double b = 1;
                double c = (a + b) / 2;
                double eps = 0.01;
                double imA = errorWeakClassifier(a, polarisation, step, featureNumber);
                if (imA <= errorWeakClassifier(a + eps, polarisation, step, featureNumber)) {
                    if (imA < errorMin) {
                        errorMin = imA;
                        polarisationOpt = polarisation;
                        thresholdOpt = a;
                    }
                }
                double imB = errorWeakClassifier(b, polarisation, step, featureNumber);
                if (imB <= errorWeakClassifier(b - eps, polarisation, step, featureNumber)) {
                    if (imB < errorMin) {
                        errorMin = imB;
                        polarisationOpt = polarisation;
                        thresholdOpt = b;
                    }
                }
                double imC = errorWeakClassifier(c, polarisation, step, featureNumber);
                for (int i = 0; i < ceil(-log(eps) / log(2) + 1); i++) {
                    if (errorWeakClassifier(c + eps, polarisation, step, featureNumber) < imC) {
                        a = c;
                    } else if (errorWeakClassifier(c - eps, polarisation, step, featureNumber) < imC) {
                        b = c;
                    } else {
                        if (imC < errorMin) {
                            errorMin = imC;
                            polarisationOpt = polarisation;
                            thresholdOpt = c;
                            break;
                        }
                    }
                    c = (a + b) / 2;
                    imC = errorWeakClassifier(b, polarisation, step, featureNumber);
                }
            }
            if (errorMinAmongFeatures > errorMin) {
                errorMinAmongFeatures = errorMin;
                polarisationOptAmongFeatures = polarisationOpt;
                thresholdOptAmongFeatures = thresholdOpt;
                optFeatureNb = featureNumber;
            }
        }
        classifiers.push_back(pair<int, double>(polarisationOptAmongFeatures, thresholdOptAmongFeatures));
        betas.push_back(errorMinAmongFeatures / (1 - errorMinAmongFeatures));
        sums[step] = 0;
        for (int i = 0; i < datasetSize; i++) {
            coef[step][i] = coef[step - 1][i];
            if ((i < positiveSet.size()
                 && weakClassifier(thresholdOptAmongFeatures, polarisationOptAmongFeatures,
                                   features[optFeatureNb].eval(positiveSet[i])) == 1) ||
                (i >= positiveSet.size()
                 && weakClassifier(thresholdOptAmongFeatures, polarisationOptAmongFeatures,
                                   features[optFeatureNb].eval(negativeSet[i - positiveSet.size()])) == 0)) {
                coef[step][i] *= betas[step];
            }
            sums[step] += coef[step][i];
        }
    }
}

void AdaBoost::print() {
    for (int i = 0; i < classifiers.size(); i++) {
        cout << "Classifieur " << i << " : [polarisation: " << classifiers[i].first << "; threshold: " <<
        classifiers[i].second << "]" << endl;
    }
}

double AdaBoost::errorWeakClassifier(double threshold, int polarisation, int step, int featureNumber) {
    double error = 0;
    for (int j = 0; j < datasetSize; j++) {
        if (j < positiveSet.size())
            error += coef[step - 1][j] *
                     (weakClassifier(threshold, polarisation, features[featureNumber].eval(positiveSet[j])) - 1);
        else
            error += coef[step - 1][j] *
                     weakClassifier(threshold, polarisation,
                                    features[featureNumber].eval(negativeSet[j - positiveSet.size()]));
    }
    return error;
}

int AdaBoost::weakClassifier(double threshold, int polarisation, double x) {
    if (polarisation * x < polarisation * threshold)
        return 1;
    else
        return 0;
}
