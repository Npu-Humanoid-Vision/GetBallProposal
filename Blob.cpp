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


// blob 在单通道意义下有意义
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

    cv::Mat used_channel;
    int used_channel_flag = 1;

    cv::SimpleBlobDetector::Params blob_params;
    // thresholds
    cv::Mat thre_result;
    int min_thre = 0;
    int max_thre = 255;
    blob_params.thresholdStep = 1;
    blob_params.minDistBetweenBlobs = 0;

	// Filter by Area.
	blob_params.filterByArea = true;
	blob_params.minArea = 10;
    blob_params.maxArea = 50000;

	// Filter by Circularity
	blob_params.filterByCircularity = false;
	blob_params.minCircularity = 0.1;

	// Filter by Convexity
	blob_params.filterByConvexity = false;
	blob_params.minConvexity = 0.87;

	// Filter by Inertia
	blob_params.filterByInertia = false;
	blob_params.minInertiaRatio = 0.01;

	// Storage for blobs
	vector<KeyPoint> keypoints;

    // Image for Showing result
    cv::Mat frame_with_blobs;

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
        cv::GaussianBlur(frame, frame, cv::Size(11, 11), 0.);

        // get used channel
        used_channel = GetUsedChannel(frame, used_channel_flag);

        // thre 
        thre_result = used_channel>min_thre & used_channel<max_thre;
        // thre_result = thre_result == 0;
        blob_params.minThreshold = min_thre*1.0;
        blob_params.maxThreshold = max_thre*1.0;
        
        // detect 
#if CV_MAJOR_VERSION < 3
        cv::SimpleBlobDetector blob_detector(blob_params);
        blob_detector.detect(used_channel, keypoints);
#else
        cv::Ptr<cv::SimpleBlobDetector> blob_detector = cv::SimpleBlobDetector::create(blob_params);
        blob_detector->detect(used_channel, keypoints);
#endif

        // draw blobs
        cv::drawKeypoints(frame, keypoints, frame_with_blobs, cv::Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::imshow("living", frame);
        cv::imshow("thre", thre_result);
        cv::imshow("blobs", frame_with_blobs);
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }
    return 0;
}
