
#include "LThreWithClf.h"

cv::VideoCapture cp(0);
cv::Mat frame;
ClfBallVision sprint_vision;
ClfBallResult gabage;
AllParameters all_p;    


// #define RUN_ON_DARWIN 

int main(int argc, char const *argv[]) {
    if (!cp.isOpened()) {
        cerr<<"open camera fail"<<endl;
        return -1;
    }

    all_p.l_min = sprint_vision.l_min;
    all_p.l_max = sprint_vision.l_max;
    all_p.a_min = sprint_vision.a_min;
    all_p.a_max = sprint_vision.a_max;
    all_p.s_min = sprint_vision.s_min;
    all_p.gaus_size = sprint_vision.gaus_size;
    all_p.verti_size = sprint_vision.verti_size;
    all_p.hori_size = sprint_vision.hori_size;

    cv::namedWindow("set_params", CV_WINDOW_NORMAL);
    
    cv::createTrackbar("l_min", "set_params", &all_p.l_min, 255);
    cv::createTrackbar("l_max", "set_params", &all_p.l_max, 255);
    cv::createTrackbar("a_min", "set_params", &all_p.a_min, 255);
    cv::createTrackbar("a_max", "set_params", &all_p.a_max, 255);
    cv::createTrackbar("s_min", "set_params", &all_p.s_min, 255);
    cv::createTrackbar("gaus_size", "set_params", &all_p.gaus_size, 66);
    cv::createTrackbar("verti_size", "set_params", &all_p.verti_size, 66);
    cv::createTrackbar("hori_size", "set_params", &all_p.hori_size, 66);

    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr<<"frame empty"<<endl;
            return -1;
        }



#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        sprint_vision.set_all_parameters(all_p);
        
        sprint_vision.imageProcess(frame, &gabage);
        
        // cv::circle(sprint_vision.src_image_, gabage.center_, 3, cv::Scalar(0, 255, 0), 3);
        // cv::rectangle(sprint_vision.src_image_, sprint_vision.result_rect_, cv::Scalar(0, 255, 0), 3);
        // if (sprint_vision.init_former_rect_) {
        //     cv::rectangle(sprint_vision.src_image_, sprint_vision.former_result_rect_, cv::Scalar(255, 0, 0), 1);
        // }        

        // cv::imshow("living", frame);
        // cv::imshow("result_show", sprint_vision. src_image_);
        // cv::imshow("threshold", sprint_vision.thresholded_image_);

        char key = cv::waitKey(100);
        if (key == 's') {
            sprint_vision.StoreParameters();
            break;
        }
        else if (key == 'q') {
            break;
        }


        // for SVM test
        // ROI = frame(ROI_Rect).clone();
        // cv::resize(ROI, ROI, cv::Size(32, 32));
        // cv::HOGDescriptor hog_des(Size(32, 32), Size(16,16), Size(8,8), Size(8,8), 9);
        // std::vector<float> hog_vec;
        // hog_des.compute(ROI, hog_vec);

        // cv::Mat t(hog_vec);
        // cv::Mat hog_vec_in_mat = t.t();
        // hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);

        // CvSVM classifier;
        // classifier.load("linear_auto.xml");

        // int lable = (int)classifier.predict(hog_vec_in_mat);
        // if (lable == POS_LABLE) {
        //     cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        // }
        // else {
        //     cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        // }
        // cv::imshow("frame", frame);
        // if (cv::waitKey(20) == 'q') {
        //     break;
        // }
    }
    return 0;
}
