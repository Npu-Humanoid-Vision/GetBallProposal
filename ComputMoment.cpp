#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define CP_OPEN "/media/alex/Data/baseRelate/code/NpuHumanoidVision/BackUpSource/BigBall/Rotated/Pos/%d.jpg"
#define CP_OPEN_NG "/media/alex/Data/baseRelate/code/NpuHumanoidVision/BackUpSource/BigBall/Rotated/Neg/%d.jpg"

double std_hu[7];
cv::Mat std_frame;

void GetHu(cv::Mat t_frame, double* hu) {
    cv::Mat frame;
    cv::cvtColor(t_frame, frame, CV_BGR2GRAY);
    cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0, 0);
    cout<<frame.size()<<endl;
    cv::Moments moment = cv::moments(frame, false); 
    cv::HuMoments(moment, hu);
    return ;
}


int main(int argc, char const *argv[]) {
    cv::VideoCapture cp(CP_OPEN);
    cv::Mat frame;

    cp>>std_frame;
    while(! std_frame.data) {
        cp>>std_frame;
        cout<<"yayaya"<<endl;
    }
    // GetHu(frame, std_hu);
    cp.open(CP_OPEN_NG);

    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cout<<"waiting for rebooting"<<endl;
            cp.open(CP_OPEN_NG);
            continue;
        }
        double hu[7];
        GetHu(frame, hu);
        cout << cv::matchShapes(std_frame, frame,CV_CONTOURS_MATCH_I1, 0.0); 
        

        cv::imshow("living", frame);
        cv::waitKey(1);
    }
    return 0;
}
