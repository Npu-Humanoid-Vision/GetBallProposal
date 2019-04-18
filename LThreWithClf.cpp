#include "Params.h"

#define CP_OPEN 0

cv::VideoCapture cp(CP_OPEN);
cv::Mat frame;

// l thre relate
cv::Mat l;
cv::Mat s;
cv::Mat thre_result;
int l_min = 0;
int l_max = 128;
int s_min = 0;

#undef MODEL_NAME
#define MODEL_NAME "../SvmTrain/model/BigBall/c_svc_with_moment&lbp.xml"


#define IMG_COLS 128
#define IMG_ROWS 128

inline cv::Mat GetHogVec(cv::Mat& ROI);

void GetProposal(std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Mat>& result_roi) {
    cout<<"contours size"<<contours.size()<<endl;
    for (size_t i=0; i<contours.size(); i++) {
        cv::Rect t_rect = boundingRect(contours[i]);
        double wh_rate = t_rect.width*1.0/t_rect.height;
 
        // shape & area thre
        if (0.5<wh_rate && wh_rate<2.0 && t_rect.area() > 200) {
            result_roi.push_back(frame(t_rect).clone());
            continue;
            // reshape rect
            if (t_rect.width < t_rect.height) {
                // reshape 之后是否超出图像边界
                if (t_rect.x + t_rect.height > frame.cols) {// 是
                    cout<<"out of image"<<endl;
                    // 够不够复制的
                    if (t_rect.x + t_rect.height/2 > frame.cols) {// 不🐶
                        continue;
                    }
                    else {// 🐶
                        cout<<"be able to flip"<<endl;
                        cout<<t_rect<<endl;
                        if (t_rect.height%2 != 0) {
                            t_rect.height -= 1;
                        }
                        cv::Mat t_roi(t_rect.height, t_rect.height, CV_8UC3);
                        // cout<<t_roi.size()<<endl;
                        cv::Rect left_half = cv::Rect(t_rect.x, t_rect.y, t_rect.height/2, t_rect.height);
                        // cv::imshow("up half", frame(up_half));
                        // cv::waitKey();
                        frame(left_half).copyTo(t_roi.colRange(0, t_rect.height/2));

                        cv::flip(frame(left_half), t_roi.colRange(t_rect.height/2, t_rect.height), -1);
                        result_roi.push_back(t_roi);
                        // cv::imshow("233", t_roi);
                        // cv::waitKey();
                    }
                }
                else {// 否
                    cout<<"not out of image"<<endl;
                    // cout<<t_rect<<endl;
                    t_rect.width = t_rect.height;

                    // cout<<t_rect<<endl;
                    // cout<<frame.size()<<endl;
                    // cv::imshow("233", frame(t_rect).clone());
                    // cv::waitKey();
                    result_roi.push_back(frame(t_rect).clone());
                }
            }
            else if (t_rect.width > t_rect.height) {

                // reshape 之后是否超出图像边界
                if (t_rect.y + t_rect.height <= frame.rows) {// 是
                    cout<<"out of image"<<endl;
                    // 够不够复制的
                    if (t_rect.y + t_rect.width/2 > frame.rows) {// 不🐶
                        continue;
                    }
                    else {// 🐶
                        cout<<"be able to flip"<<endl;
                        cout<<t_rect<<endl;
                        if (t_rect.width%2 != 0) {
                            t_rect.width -= 1;
                        }
                        cv::Mat t_roi(t_rect.width, t_rect.width, CV_8UC3);
                        // cout<<t_roi.size()<<endl;
                        cv::Rect up_half = cv::Rect(t_rect.x, t_rect.y, t_rect.width, t_rect.width/2);
                        // cv::imshow("up half", frame(up_half));
                        // cv::waitKey();
                        frame(up_half).copyTo(t_roi.rowRange(0, t_rect.width/2));

                        cv::flip(frame(up_half), t_roi.rowRange(t_rect.width/2, t_rect.width), -1);

                        result_roi.push_back(t_roi);
                        // cv::imshow("233", t_roi);
                        // cv::waitKey();
                    }
                }
                else {// 否
                    cout<<"not out of image"<<endl;
                    // cout<<t_rect<<endl;
                    t_rect.height = t_rect.width;

                    // cout<<t_rect<<endl;
                    // cout<<frame.size()<<endl;
                    // cv::imshow("233", frame(t_rect).clone());
                    // cv::waitKey();
                    result_roi.push_back(frame(t_rect).clone());
                }

            }
            else {
                // cv::imshow("233", frame(t_rect).clone());
                // cv::waitKey();
                cout<<"width equals height"<<endl;
                result_roi.push_back(frame(t_rect).clone());
            }

        }
    }
    cout<<"proposal size: "<<result_roi.size()<<endl;
}
void GetProposal_v2(std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Rect>& result_roi) {
    for (size_t i=0; i<contours.size(); i++) {
        cv::Rect t_rect = boundingRect(contours[i]);
        double wh_rate = t_rect.width*1.0/t_rect.height;
 
        // shape & area thre
        if (0.5<wh_rate && wh_rate<2.0 && t_rect.area() > 200) {
            if (t_rect.width < t_rect.height) {
                continue;
            }
            else if (t_rect.width > t_rect.height) {
                int delta = t_rect.width/18;

                if (t_rect.x-delta > 0) 
                    t_rect.x -= delta;
                if (t_rect.y-delta >0)
                    t_rect.y -= delta;

                if (t_rect.y + t_rect.width < frame.rows)
                    t_rect.height = t_rect.width;   
                if (t_rect.width+t_rect.x+delta < frame.cols)
                    t_rect.width += delta;
                if (t_rect.height+t_rect.y+delta < frame.rows)
                    t_rect.height += delta;
                cout<<t_rect<<endl;
                result_roi.push_back(t_rect);
            }
            else {
                cout<<t_rect<<endl;
                result_roi.push_back(t_rect);
            }

        }
    }
}


void on_trackbar(int, void*) {
    l = GetUsedChannel(frame, L);
    s = GetUsedChannel(frame, S);
    thre_result = (l_min<=l & l<=l_max) & s>=s_min;
    cv::imshow("thre result", thre_result);
}

int main() {
    cout<<MODEL_NAME<<endl;
    cv::namedWindow("params");
    cv::createTrackbar("min_l", "params", &l_min, 256, on_trackbar);
    cv::createTrackbar("max_l", "params", &l_max, 256, on_trackbar);
    cv::createTrackbar("min_s", "params", &s_min, 256, on_trackbar);

    cv::namedWindow("living");
    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cout<<"rebooting cam..."<<endl;
            cp.open(CP_OPEN);
            continue;
        }
        cv::Mat raw_frame = frame.clone();
        cv::GaussianBlur(frame, frame, cv::Size(11, 11), 0);

        l = GetUsedChannel(frame, L);
        s = GetUsedChannel(frame, S);

        thre_result = (l_min<=l & l<=l_max) & s>=s_min;

        // cv::erode(thre_result, thre_result, cv::Mat(5, 5, CV_8UC1));
        // cv::dilate(thre_result, thre_result, cv::Mat(5, 5, CV_8UC1));

        std::vector<std::vector<cv::Point>> contours;
        cv::Mat ttt = thre_result.clone();
        cv::findContours(ttt, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        cout<<"contours size: "<<contours.size()<<endl;
        std::vector<cv::Mat> proposal;
        std::vector<cv::Rect> proposal_rect;
        GetProposal_v2(contours, proposal_rect);
        cout<<"proposal size: "<<proposal_rect.size()<<endl;
        for (auto i=proposal_rect.begin(); i!=proposal_rect.end(); i++) {
            cv::Mat roi = raw_frame(*i).clone();
            cv::Mat hog_vec_in_mat = GetHogVec(roi);
#if CV_MAJOR_VERSION < 3
            CvSVM tester;
            tester.load(MODEL_NAME);
            int lable = (int)tester.predict(hog_vec_in_mat);
            if (lable == POS_LABLE) {
                cv::rectangle(frame, *i, cv::Scalar(0, 255, 0), 2);
            }
            else {
                cv::rectangle(frame, *i, cv::Scalar(0, 0, 255), 2);
            }
#else
            cv::Ptr<cv::ml::SVM> tester = cv::ml::SVM::load(MODEL_NAME);
            cv::Mat lable;
            tester->predict(hog_vec_in_mat, lable);
            if (lable.at<float>(0, 0) == POS_LABLE) {
                cv::rectangle(frame, *i, cv::Scalar(0, 255, 0), 2);
            } 
            else {
                cv::rectangle(frame, *i, cv::Scalar(0, 0, 255), 2);
            }
#endif

            // cout<<"show"<<endl;
            // cv::imshow("yayaya", *i);
            // cv::waitKey();
        }

        cv::imshow("thre result", thre_result);
        cv::imshow("living", frame);
        char key = cv::waitKey(200);
        if (key == 'q') {
            break;
        }
    }
    return 0;
}


inline cv::Mat GetHogVec(cv::Mat& ROI) {
    cv::resize(ROI, ROI, cv::Size(IMG_COLS, IMG_ROWS));
    cv::HOGDescriptor hog_des(Size(IMG_COLS, IMG_ROWS), Size(16,16), Size(8,8), Size(8,8), 9);
    std::vector<float> hog_vec;
    hog_des.compute(ROI, hog_vec);

    for (int j=0; j<6; j++) {
        cv::Mat ROI_l = GetUsedChannel(ROI, j);
        cv::Moments moment = cv::moments(ROI_l, false);


        // lbp related
        cv::Mat lbp_mat;
        // cv::resize(t_image_l, t_image_l, cv::Size(30, 30));
        calExtendLBPFeature(ROI_l, Size(16, 16), lbp_mat);
        for (int k=0; k<lbp_mat.cols; k++) {
            hog_vec.push_back(lbp_mat.at<float>(0, k));
        }

        double hu[7];
        cv::HuMoments(moment, hu);
        for (int k=0; k<7; k++) {
            hog_vec.push_back(hu[k]);
        }
        // for (int k=0; k<lbp_vec.cols; k++) {
        //     t_descrip_vec.push_back(lbp_vec.at<uchar>(0, k));
        // }
    }

    cv::Mat t(hog_vec);
    cv::Mat hog_vec_in_mat = t.t();
    hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);

    return hog_vec_in_mat;
}