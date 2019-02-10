#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


int main(int argc, char const *argv[]) {
    // cv::VideoCapture cp("D:/baseRelate/code/svm_trial/BackUpSource/Ball/Train/Raw/%d.jpg");
    cv::VideoCapture cp(0);
    cv::Mat frame;
    cv::Mat blured_frame;

    // 高斯模糊相关
    int gauss_kernal_size = 2;
    cv::namedWindow("blured");
    cv::createTrackbar("gauss_kernal_size", "blured", &gauss_kernal_size, 9);

    // L thre
    int L_min = 0, L_max = 255;
    cv::namedWindow("L");
    cv::createTrackbar("L_min", "L", &L_min, 255);
    cv::createTrackbar("L_max", "L", &L_max, 255);

    // H thre
    int H_min = 0, H_max = 255;
    cv::namedWindow("H");
    cv::createTrackbar("H_min", "H", &H_min, 255);
    cv::createTrackbar("H_max", "H", &H_max, 255);

    // S thre
    int S_min = 0, S_max = 255;
    cv::namedWindow("S");
    cv::createTrackbar("S_min", "S", &S_min, 255);
    cv::createTrackbar("S_max", "S", &S_max, 255);

    int porc_flag = 0;

    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr << "Line: " << __LINE__ << endl;
            // return -1;
            continue;
        }
        cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));

        cv::GaussianBlur(frame, blured_frame, cv::Size(2*gauss_kernal_size+1, 2*gauss_kernal_size+1), 0., 0.);
        cv::imshow("blured", blured_frame);

        // H, L, S 通道
        cv::Mat t_HLS;
        cv::Mat t_HLS_Channels[3];
        cv::cvtColor(frame, t_HLS, CV_BGR2HLS_FULL);
        cv::split(t_HLS, t_HLS_Channels);

        cv::Mat h_thre_frame = t_HLS_Channels[0] >= H_min & t_HLS_Channels[0] <= H_max;
        cv::erode(h_thre_frame, h_thre_frame, cv::Mat(5, 5, CV_8UC1));
        cv::dilate(h_thre_frame, h_thre_frame, cv::Mat(5, 5, CV_8UC1));

        cv::imshow("L", t_HLS_Channels[1] >= L_min & t_HLS_Channels[1] <= L_max);
        cv::imshow("H", h_thre_frame);
        cv::imshow("S", t_HLS_Channels[2] >= S_min & t_HLS_Channels[2] <= S_max);

        if (porc_flag) {

            std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::Mat ttttt = t_HLS_Channels[0] >= H_min & t_HLS_Channels[0] <= H_max;
            cv::findContours( ttttt, 
                        contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            cv::Mat contours_image = cv::Mat::zeros(frame.size(), CV_8UC1);
            for (auto i = 0; i < contours.size(); i++) {
                cv::drawContours(contours_image, contours, i, cv::Scalar(255));
            }
            cv::imshow("contours", contours_image);

            double max_area = -1.;
            std::vector<cv::Point> max_contours;
            for (auto i = contours.begin(); i != contours.end(); i++) {
                if (contourArea(*i) > max_area) {
                    max_contours = *i;
                }
            }

            cv::Mat ball_candicate = cv::Mat::zeros(frame.size(), CV_8UC1);
            cv::Mat L_thre_result = t_HLS_Channels[1] >= L_min & t_HLS_Channels[1] <= L_max;

            cv::Mat_<uchar>::iterator begin = L_thre_result.begin<uchar>();
            cv::Mat_<uchar>::iterator end = L_thre_result.end<uchar>();
            while (begin != end) {
                // cout<<begin.pos()<<endl;
                if (*begin == 255 && cv::pointPolygonTest(max_contours, begin.pos(), false) == 1) { // 最后一个此处必须为 false 否则返回 distance
                    ball_candicate.at<uchar>(begin.pos()) = 255;
                    
                }
                begin++;
            }
            cv::imshow("ball_candidate", ball_candicate);

        }
        cv::imshow("living", frame);
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
        else if (key == 'b') {
            porc_flag = ~porc_flag;
            cout<<porc_flag<<endl;
        }
    }
    return 0;
}
