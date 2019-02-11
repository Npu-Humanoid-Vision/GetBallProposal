#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define POS_LABLE 1
#define NEG_LABLE 0

// VideoCapture打开的东西(string& filename/webcam index)
// #define CP_OPEN "/media/alex/Data/baseRelate/pic_data/frame%04d.jpg"
#define CP_OPEN "/media/alex/Data/baseRelate/code/NpuHumanoidVision/BackUpSource/Ball/Train/Raw/%d.jpg"

#define MODEL_NAME "../SvmTrain/ball_linear_auto.xml"

#define IMG_COLS 32
#define IMG_ROWS 32




inline cv::Mat GetUsedChannel(cv::Mat& image, int flag) {
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

inline void Slide(cv::Mat& integral_image, std::vector<cv::Rect>& result, double thre) {
    // define the wins size
    std::vector<cv::Size> wins_sizes;
    for (int i=10; i<=100; i+=10) {
        wins_sizes.push_back(cv::Size(i,i));
    }
    int row = integral_image.rows;
    int col = integral_image.cols;
    
    int row_step = 10;
    int col_step = 10;
    for (int k=0; k<wins_sizes.size(); k++) {
        for (int i=0; i+wins_sizes[k].height<row; i+=row_step) {
            for (int j=0; j+wins_sizes[k].width<col; j+=col_step) {
                // compute ratio for thre
                int win_sum = integral_image.at<int>(i+wins_sizes[k].height, j+wins_sizes[k].width) 
                            + integral_image.at<int>(i, j)
                            - integral_image.at<int>(i+wins_sizes[k].height, j)
                            - integral_image.at<int>(i, j+wins_sizes[k].width);
                if (1.0*win_sum/wins_sizes[k].area() > thre) {
                    result.push_back(cv::Rect(cv::Point(j, i), wins_sizes[k]));
                }    
            }
        }
    }  
}

inline cv::Mat GetHogVec(cv::Mat& ROI) {
    cv::resize(ROI, ROI, cv::Size(IMG_COLS, IMG_ROWS));
    cv::HOGDescriptor hog_des(Size(IMG_COLS, IMG_ROWS), Size(16,16), Size(8,8), Size(8,8), 9);
    std::vector<float> hog_vec;
    hog_des.compute(ROI, hog_vec);

    cv::Mat t(hog_vec);
    cv::Mat hog_vec_in_mat = t.t();
    hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);

    return hog_vec_in_mat;
}

int main() {
    // load SVM model
#if CV_MAJOR_VERSION < 3
    CvSVM tester;
    tester.load(MODEL_NAME);
#else
    cv::Ptr<cv::ml::SVM> tester = cv::ml::SVM::load(MODEL_NAME);
#endif

    cv::VideoCapture cp(CP_OPEN);
    cv::Mat frame;
    cv::Mat integral_frame;
    cv::Mat used_channel;
    cv::Mat probable_pos;
    int used_channel_flag = 1;

    // thresholds
    cv::Mat thre_result;
    int min_thre = 0;
    int max_thre = 255;
    
    // fps variables
    double begin;
    double fps;

    cv::namedWindow("blob_params");
    cv::createTrackbar("min_thre", "blob_params", &min_thre, 256);
    cv::createTrackbar("max_thre", "blob_params", &max_thre, 256);
    while (true) {
        begin = (double)getTickCount();

        cp >> frame;
        if (frame.empty()) {
            cout<<"wait for rebooting..."<<endl;
            cp.open(CP_OPEN);
            continue;
        }
        
#if CV_MAJOR_VERSION < 3
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        // blur 
        cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0.);

        // get used channel 
        used_channel = GetUsedChannel(frame, used_channel_flag);

         // thre 
        thre_result = used_channel>=min_thre & used_channel<=max_thre;

        // get intergral image
        cv::integral(thre_result, integral_frame, CV_32S);
        integral_frame /= 255;

        std::vector<cv::Rect> sld_result;
        Slide(integral_frame, sld_result, 0.5);
        
        probable_pos = frame.clone();
        for (auto i = sld_result.begin(); i != sld_result.end(); i++) {
            cv::Mat t = frame(*i).clone();
            cv::Mat hog_vec_in_mat = GetHogVec(t);
#if CV_MAJOR_VERSION < 3
            int lable = (int)tester.predict(hog_vec_in_mat);
            if (lable == POS_LABLE) {
                cv::rectangle(frame, *i, cv::Scalar(0, 255, 0), 2);
            }
            else {
                cv::rectangle(frame, *i, cv::Scalar(0, 0, 255), 2);
            }
#else
            cv::Mat lable;
            tester->predict(hog_vec_in_mat, lable);
            if (lable.at<float>(0, 0) == POS_LABLE) {
                cv::rectangle(frame, *i, cv::Scalar(0, 255, 0), 2);
            } 
            else {
                // cv::rectangle(frame, *i, cv::Scalar(0, 0, 255), 2);
            }
#endif
        }
        cout<<"fps: "<<1.0/(((double)getTickCount() - begin)/getTickFrequency())<<endl;

        cv::imshow("living", frame);
        cv::imshow("thre", thre_result);
        // cv::imshow("integral", integral_frame);
        cv::imshow("sld_result", probable_pos);
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }
    return 0;
}