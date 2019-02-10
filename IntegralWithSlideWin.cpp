#include <bits/stdc++.h>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

// VideoCapture打开的东西(string& filename/webcam index)
#define CP_OPEN "/media/alex/Data/baseRelate/pic_data/frame%04d.jpg"

cv::Mat GetUsedChannel(cv::Mat& image, int flag) {
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

int main() {
    cv::VideoCapture cp(CP_OPEN);
    cv::Mat frame;
    cv::Mat integral_frame;
    cv::Mat used_channel;
    int used_channel_flag = 1;

    // thresholds
    cv::Mat thre_result;
    int min_thre = 0;
    int max_thre = 255;
    cv::namedWindow("blob_params");
    cv::createTrackbar("min_thre", "blob_params", &min_thre, 256);
    cv::createTrackbar("max_thre", "blob_params", &max_thre, 256);
    while (true) {
        cp >> frame;
        if (frame.empty()) {
            cout<<"wait for cam init..."<<endl;
            continue;
        }
        // blur 
        cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0.);

        // get used channel
        used_channel = GetUsedChannel(frame, used_channel_flag);

         // thre 
        thre_result = used_channel>min_thre & used_channel<max_thre;

        // get intergral image
        cv::integral(thre_result, integral_frame, CV_32S);
        cv::normalize(integral_frame, integral_frame, 0, 255, CV_MINMAX);
        cv::convertScaleAbs(integral_frame, integral_frame);

        cv::imshow("living", frame);
        cv::imshow("thre", thre_result);
        cv::imshow("integral", integral_frame);
        char key = cv::waitKey(0);
        if (key == 'q') {
            break;
        }
    }
    return 0;
}