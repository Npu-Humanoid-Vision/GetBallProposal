#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define POS_LABLE 1
#define NEG_LABLE 0

// VideoCapture打开的东西(string& filename/webcam index)
// #define CP_OPEN "/media/alex/Data/baseRelate/pic_data/frame%04d.jpg"
#define CP_OPEN "/media/alex/Data/baseRelate/code/NpuHumanoidVision/BackUpSource/Ball/Train/Raw/%d.jpg"
// #define CP_OPEN 0s

#define MODEL_NAME "../SvmTrain/ball_rbf_auto.xml"

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
    for (int i=100; i<=300; i+=40) {
        wins_sizes.push_back(cv::Size(i,i));
    }
    int row = integral_image.rows;
    int col = integral_image.cols;
    
    int row_step = 40;
    int col_step = 40;
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

// fit the line by the y = k*x + b form
inline void GetSideLine(cv::Mat& binary_image, double& k, double& b) {
    // points for sideline
    std::vector<cv::Point2i> sideline_points; 

    // integral the binary image for latter operation
    cv::Mat integral_image;
    cv::Mat t = binary_image/255;
    cv::integral(t, integral_image, CV_32S);

    // slide wins related validables
    int wins_stride = 10;
    int wins_num = 32;
    int wins_row = 10;
    double wins_thre = 0.81;
    int wins_col = binary_image.cols/wins_num;
    std::vector<cv::Rect> wins;

    for (auto i = 0; i < wins_num; i++) {
        wins.push_back(cv::Rect(i*wins_row, 0, wins_col, wins_row));

        int pix_counter = 0;
        cv::Rect& t_rect = wins[i];
        do {
            pix_counter = integral_image.at<int>(t_rect.y+t_rect.height, t_rect.x+t_rect.width)
                        + integral_image.at<int>(t_rect.y, t_rect.x)
                        - integral_image.at<int>(t_rect.y+t_rect.height, t_rect.x)
                        - integral_image.at<int>(t_rect.y, t_rect.x+t_rect.width);
            if (1.0*pix_counter/t_rect.area() > wins_thre) {
                break;
            }

            if (t_rect.y+2*wins_stride < binary_image.rows) {
                t_rect.y += wins_stride;
            }
            else {
                break;
            }
        }while (true);
        sideline_points.push_back(cv::Point2i(t_rect.x + wins_col/2, t_rect.y + wins_row*wins_thre));
    }

    // fit the sideline discrete points by least quares method
    cv::Mat mat_a(wins_num, 2, CV_64FC1);
    cv::Mat mat_x(2, 1, CV_64FC1);
    cv::Mat mat_b(wins_num, 1, CV_64FC1);

    for (int i = 0; i < wins_num; i++) {
        mat_a.at<double>(i, 0) = sideline_points[i].x;
        mat_a.at<double>(i, 1) = 1.;

        mat_b.at<double>(i, 0) = sideline_points[i].y;
    }
    cv::Mat mat_a_t = mat_a.t();
    mat_x = (mat_a_t*mat_a).inv(DECOMP_LU)*mat_a_t*mat_b;

    // return k and b
    k = mat_x.at<double>(0, 0);
    b = mat_x.at<double>(1, 0);
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
        char key = cv::waitKey();
        if (key == 'q') {
            break;
        }
    }
    return 0;
}