#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define CP_OPEN 1
// #define CP_OPEN "/media/alex/Data/baseRelate/pic_data/frame%04d.jpg"


enum { H,S,V,L,A,B };

// L thre
int L_min = 0, L_max = 255;

cv::Mat GetUsedChannel(cv::Mat& src_img, int flag);

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


cv::Mat GetUsedChannel(cv::Mat& src_img, int flag) {
    cv::Mat t;
    cv::Mat t_cs[3];
    switch (flag) {
    case 0:
    case 1:
    case 2:
        cv::cvtColor(src_img, t, CV_BGR2HSV_FULL);
        cv::split(t, t_cs);
        return t_cs[flag];
    case 3:
    case 4:
    case 5:
        cv::cvtColor(src_img, t, CV_BGR2Lab);
        cv::split(t, t_cs);
        return t_cs[flag - 3];
    }
}

void GetGrassEdge(cv::Mat& binary_image, cv::Mat& edge_image) {
    edge_image = cv::Mat(binary_image.size(), CV_8UC1, cv::Scalar(0));
    int last_i = -1;
    int last_j = -1;
    for (int i = 0; i < binary_image.cols; i++) {
        for (int j = 0; j < binary_image.rows-3; j++) {
            if (binary_image.at<uchar>(j, i) == 255
                && binary_image.at<uchar>(j+1, i) == 255
                && binary_image.at<uchar>(j+2, i) == 255
                && binary_image.at<uchar>(j+3, i) == 255) {
                if (last_i < 0) {
                    last_i = i;
                    last_j = j;
                } else {
                    cv::line(edge_image, cv::Point(i, j), cv::Point(last_i, last_j), cv::Scalar(255), 2);
                    last_i = i;
                    last_j = j;
                }
                break;
            }
        }
    }
    cv::imshow("233", edge_image);
    cv::floodFill(edge_image, cv::Point(binary_image.cols / 2, binary_image.rows - 1), 255);
}

void JudgeRoi(cv::Mat binary_image, std::vector<std::vector<cv::Point>>& judeged_contours) {
    std::vector<std::vector<cv::Point>> contours;

    cv::Mat ttt = binary_image.clone();

    cv::findContours(ttt, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++) {
        int min_x = 666;
        int max_x = -1;
        int min_y = 666;
        int max_y = -1;

        for (int j = 0; j < contours[i].size(); j++) {
            if (contours[i][j].x < min_x) {
                min_x = contours[i][j].x;
            }
            if (contours[i][j].x > max_x) {
                max_x = contours[i][j].x;
            }
            if (contours[i][j].y < min_y) {
                min_y = contours[i][j].y;
            }
            if (contours[i][j].y > max_y) {
                max_y = contours[i][j].y;
            }
        }
        int delta_x = max_x - min_x;
        int delta_y = max_y - min_y;

        if (0.5 < delta_x * 1.0 / delta_y && delta_x * 1.0 / delta_y < 2 && delta_x*delta_y > 200) {
            judeged_contours.push_back(contours[i]);
        }
    }
}



int main(int argc, char const* argv[]) {
    cv::Mat std_frame = cv::imread("/media/alex/Data/baseRelate/code/NpuHumanoidVision/BackUpSource/BigBall/Rotated/Pos/0.jpg");
    cv::Mat std_binary;
    L_min = 140;
    std::vector<cv::Point> std_contour = GetMaxContours(std_frame, std_binary);

    cv::VideoCapture cp(CP_OPEN);
    cv::Mat frame;
    cv::namedWindow("params");

    // Channel thre relate
    int l_min = 180;
    int l_max = 255;
    cv::createTrackbar("l_min", "params", &l_min, 255);
    cv::createTrackbar("l_max", "params", &l_max, 255);

    int black_max = 10;
    cv::createTrackbar("black max", "params", &black_max, 255);

    int a_min = 62;
    int a_max = 125;
    cv::createTrackbar("a_min", "params", &a_min, 255);
    cv::createTrackbar("a_max", "params", &a_max, 255);

    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cout << "waiting for camera rebooting" << endl;
            cp.open(CP_OPEN);
            continue;
        }
        cv::Mat gaused_frame;
        cv::GaussianBlur(frame, gaused_frame, cv::Size(11, 11), 0);

        cv::Mat l = GetUsedChannel(gaused_frame, L);
        cv::Mat a = GetUsedChannel(gaused_frame, A);

        cv::Mat l_thre = (l_min <= l & l <= l_max) | l < black_max;
        cv::Mat a_thre = a_min < a & a < a_max;

        cv::dilate(l_thre, l_thre, cv::Mat(5, 5, CV_8UC1));
        cv::erode(l_thre, l_thre, cv::Mat(5, 5, CV_8UC1));

        cv::dilate(a_thre, a_thre, cv::Mat(5, 5, CV_8UC1));
        cv::erode(a_thre, a_thre, cv::Mat(5, 5, CV_8UC1));

        cv::Mat a_thre_flood;
        GetGrassEdge(a_thre, a_thre_flood);

        cv::Mat l_and_a = l_thre & a_thre_flood;

        cv::dilate(l_and_a, l_and_a, cv::Mat(5, 5, CV_8UC1));
        cv::erode(l_and_a, l_and_a, cv::Mat(5, 5, CV_8UC1));

        std::vector<std::vector<cv::Point>> contours;
        JudgeRoi(l_and_a, contours);

        std::vector<std::vector<cv::Point>> contours_poly(contours.size());
        std::vector<cv::Rect> boundRect(contours.size());
        std::vector<cv::Point2f> center(contours.size());
        std::vector<float> radius(contours.size());

        for (int i = 0; i < contours.size(); i++) {
            approxPolyDP(contours[i], contours_poly[i], 3, true);
            boundRect[i] = boundingRect(contours_poly[i]);
            minEnclosingCircle(contours_poly[i], center[i], radius[i]);
            // rectangle(frame, boundRect[i].tl(), boundRect[i].br(), (0, 0, 255), 2, 8, 0);
        }

        for (int i = 0; i < contours.size(); i++) {
            cv::Mat roi = a_thre(boundRect[i]);
            // cout<<1.0*sum(roi)[0]/255.0/boundRect[i].area()<<endl;
            if (1.0*sum(roi)[0]/255.0/boundRect[i].area() > 0.8) {
                continue;
            }
            else {
                rectangle(frame, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0, 255, 0), 2, 8, 0);
            }
        }

        int min_score_idx;
        double min_score = 100000;
        for (int i=0; i<contours.size(); i++) {
            double t_score = matchShapes(contours[i], std_contour, CV_CONTOURS_MATCH_I1, 0.);
            if (t_score < min_score) {
                min_score = t_score;
                min_score_idx = i;
            }
        }
        cout<<"min score"<<min_score<<endl;
        cv::rectangle(frame, boundRect[min_score_idx].tl(), boundRect[min_score_idx].br(), cv::Scalar(255, 100, 0), 2, 8, 0);

        cv::imshow("src", frame);
        cv::imshow("gaus", gaused_frame);
        cv::imshow("l thre", l_thre);
        cv::imshow("a thre", a_thre);
        cv::imshow("a thre flood", a_thre_flood);
        cv::imshow("a and l thre", l_and_a);
        // cv::imshow("a mor grad", a_thre_mor_gradiant);
        char key = cv::waitKey(0);
        if (key == 'q') {
            break;
        }
    }
    return 0;
}
