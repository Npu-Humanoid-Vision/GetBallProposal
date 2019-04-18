// this version is using the built-in function in the hog classes
// so ATTENTION !!!!!
// LINEAR SVM CLASSIFIER ONLY !!!!!
#include "Params.h"

#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// 是否在达尔文上跑
// 可用 CV_MAJOR_VERSION代替
// #define RUN_ON_DARWIN

#define CP_OPEN 1

// #define MODEL_NAME "../SvmTrain/ball_linear_auto.xml"

class MySVM: public CvSVM
{
public:
    double * get_alpha_data()
    {
        return this->decision_func->alpha;
    }
    double  get_rho_data()
    {
        return this->decision_func->rho;
    }
};

inline void GetDetector(std::vector<float>& myDetector) {
    MySVM SVM;
    SVM.load(MODEL_NAME);
    

    int descriptorDim = SVM.get_var_count();
    int supportVectorNum = SVM.get_support_vector_count();
    cout<<"support vector num: "<< supportVectorNum <<endl;

    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
    Mat supportVectorMat = Mat::zeros(supportVectorNum, descriptorDim, CV_32FC1);
    Mat resultMat = Mat::zeros(1, descriptorDim, CV_32FC1);

    for (int i = 0; i < supportVectorNum; i++)//复制支持向量矩阵
    {
        const float * pSupportVectorData = SVM.get_support_vector(i);
        for(int j = 0 ;j < descriptorDim; j++)
        {
            supportVectorMat.at<float>(i,j) = pSupportVectorData[j];
        }
    }

    double *pAlphaData = SVM.get_alpha_data();
    for (int i = 0; i < supportVectorNum; i++)//复制函数中的alpha 记住决策公式Y= wx+b
    {
        alphaMat.at<float>(0, i) = pAlphaData[i];
    }

    resultMat = -1 * alphaMat * supportVectorMat; //alphaMat就是权重向量

    //cout<<resultMat;

    cout<<"descriptorDim: "<<descriptorDim<<endl;
    for (int i = 0 ;i < descriptorDim; i++)
    {
        myDetector.push_back(resultMat.at<float>(0, i));
    }

    float rho = SVM.get_rho_data();
    myDetector.push_back(rho);
    cout<<"myDetector.size: "<<myDetector.size()<<endl;
}

int main() {
    // fps variables
    double begin;
    double fps;


    cv::VideoCapture cp(CP_OPEN);
    cv::Mat frame;

    cv::HOGDescriptor hog(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    std::vector<float> detector;
    GetDetector(detector);
    hog.setSVMDetector(detector);

    std::vector<cv::Rect> result;
    cp >> frame;
    while (frame.empty()) {
        cp >> frame;
    }
    while (1) {
        begin = (double)getTickCount();
        cp >> frame;
        if (frame.empty()) {
            cerr << __LINE__ <<"frame empty"<<endl;
            return -1;
        }
#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        hog.detectMultiScale(frame, result);
        
        for (auto i = result.begin(); i != result.end(); i++) {
            cv::rectangle(frame, *i, cv::Scalar(0, 255, 0));
        }
        cv::imshow("233", frame);
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
        cout<<"fps: "<<1.0/(((double)getTickCount() - begin)/getTickFrequency())<<endl;
    }
}
// https://www.cnblogs.com/louyihang-loves-baiyan/p/4658478.html