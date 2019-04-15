#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define CP_OPEN "/media/alex/Data/baseRelate/code/NpuHumanoidVision/BackUpSource/BigBall/Rotated/Pos/%d.jpg"
// #define CP_OPEN_NG "/media/alex/Data/baseRelate/code/NpuHumanoidVision/BackUpSource/BigBall/Rotated/Neg/%d.jpg"
#define CP_OPEN_NG "/media/alex/Data/baseRelate/code/NpuHumanoidVision/BackUpSource/BigBall/Rotated/Pos/%d.jpg"


cv::Mat std_frame;
cv::Mat std_binary;
std::vector<cv::Point> std_contour;

cv::VideoCapture cp(CP_OPEN);
cv::Mat frame;
cv::Mat frame_binary;

enum {H, L, S, V, GRAY};
cv::Mat GetUsedChannel(cv::Mat image, int flag) {
    cv::Mat hls_image;
    cv::Mat hsv_image;
    cv::Mat t_cs[3];

    switch (flag) {
    case 0:// H channel
    case 1:// L channel
    case 2:// S channel
        cv::cvtColor(image, hls_image, CV_BGR2HLS_FULL);
        cv::split(hls_image, t_cs);
        return t_cs[flag];
    case 3:// V channel
        cv::cvtColor(image, hsv_image, CV_BGR2HSV_FULL);
        cv::split(hsv_image, t_cs);
        return t_cs[2];
    case 4:// gray channel
        cv::cvtColor(image, t_cs[0], CV_BGR2GRAY);
        return t_cs[0];
    }
}

// L thre
int L_min = 0, L_max = 255;


std::vector<cv::Point> GetMaxContours(cv::Mat frame, cv::Mat& binary_image) {

    std::vector<std::vector<cv::Point>> contours;
    frame = GetUsedChannel(frame, L);
    binary_image = L_min < frame & frame <= L_max;
    cv::findContours(binary_image.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    int max_idx = -1;
    int max_area = -1;
    for(unsigned int i = 0; i < contours.size(); i++ ) {
        if (contourArea(contours[i]) > max_area) {
            max_idx = i;
            max_area = contourArea(contours[i]);
        }
    }
    return contours[max_idx];
}
void on_trackbar(int, void*) {
    std::vector<cv::Point> max_contour = GetMaxContours(frame, frame_binary);
    cv::imshow("std", std_binary);
    cv::imshow("sample", frame_binary);
    cv::imshow("living", frame);

    cout<<matchShapes(max_contour, std_contour, CV_CONTOURS_MATCH_I1, 0.)<<endl;
}

int main(int argc, char const *argv[]) {

    cv::namedWindow("L");
    cv::createTrackbar("L_min", "L", &L_min, 255, on_trackbar);
    cv::createTrackbar("L_max", "L", &L_max, 255, on_trackbar);

    cp>>std_frame;
    while(std_frame.empty()) {
        cp>>std_frame;
        cout<<"yayaya"<<endl;
    }
    L_min = 140;
    std_contour = GetMaxContours(std_frame, std_binary);

    cp.open(CP_OPEN_NG);
    
    while (1) {  
        cp >> frame;
        if (frame.empty()) {
            cout<<"waiting for rebooting"<<endl;
            cp.open(CP_OPEN_NG);
            continue;
        }
        std::vector<cv::Point> max_contour = GetMaxContours(frame, frame_binary);
        

        cout<<matchShapes(max_contour, std_contour, CV_CONTOURS_MATCH_I1, 0.)<<endl;
        cv::imshow("std", std_binary);
        cv::imshow("sample", frame_binary);
        cv::imshow("living", frame);
        cv::waitKey(0);
    }
    return 0;
}
