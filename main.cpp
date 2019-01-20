#include "HogGetter.h"

int main(int argc, char const *argv[]) {
    cv::VideoCapture cp("D:/baseRelate/code/svm_trial/BackUpSource/Ball/Train/Raw/%d.jpg");
    cv::Mat image;
    while (1) {
        cp >> image;
        if (image.empty()) {
            cerr << "Line: " << __LINE__ << endl;
            return -1;
        }
        


        cv::imshow("233", image);
        cv::waitKey();
    }
    return 0;
}
