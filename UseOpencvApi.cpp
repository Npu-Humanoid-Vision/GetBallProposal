// this version is using the built-in function in the hog classes
// so ATTENTION !!!!!
// LINEAR SVM CLASSIFIER ONLY !!!!!



#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// 是否在达尔文上跑
// 可用 CV_MAJOR_VERSION代替
// #define RUN_ON_DARWIN

#define CP_OPEN 1

int main() {
    cv::VideoCapture cp(CP_OPEN);
    cv::Mat frame;

    cv::HOGDescriptor hog(Size(32, 32), Size(16, 16), Size(2, 2), Size(8, 8), 12);


    cp >> frame;
    while (frame.empty()) {
        cp >> frame;
    }
    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr << __LINE__ <<"frame empty"<<endl;
            return -1;
        }
#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif

    }
}