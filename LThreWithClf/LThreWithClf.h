#ifndef L_THRE_WITH_CLF_H
#define L_THRE_WITH_CLF_H

// 调参开关
#define ADJUST_PARAMETER


// 正负样本的 lable
#define POS_LABLE 1
#define NEG_LABLE 0


#include <opencv2/opencv.hpp>
#include <fstream> // 推荐在基类加...
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

#ifdef ADJUST_PARAMETER

// showing image in debugging 
#define SHOW_IMAGE(imgName, win_name) \
    namedWindow(win_name, WINDOW_AUTOSIZE); \
    imshow(win_name, imgName); \

class ImgProcResult{

public:
	ImgProcResult(){};
	~ImgProcResult(){};
	 virtual void operator=(ImgProcResult &res) = 0;
private:
protected:

};
class ImgProc{

public:
	ImgProc(){};
	~ImgProc(){};
	virtual void imageProcess(cv::Mat img, ImgProcResult *Result) =0;
private:
protected:
	ImgProcResult *res;

};
#else

#include "imgproc.h"
#define SHOW_IMAGE(imgName) ;

#endif 


enum { H,S,V,L,A,B };

// adjust_parameter
class ClfBallResult : public ImgProcResult
{
public:
    cv::Point center_;
    cv::Rect bound_box_;
    bool valid_;
public:
    ClfBallResult() {
        valid_ = false;
    }

    // adjust_parameter    
    virtual void operator=(ImgProcResult &res) {
        ClfBallResult *tmp = dynamic_cast<ClfBallResult *>(&res);
        center_ = tmp->center_;
        valid_ = tmp->valid_;
        bound_box_ = tmp->bound_box_;
    }

    void operator=(ClfBallResult &res) {
        center_ = res.center_;
        valid_ = res.valid_;
        bound_box_ = res.bound_box_;
    }
};

struct AllParameters {
    int l_min;
    int l_max;
    int s_min;
    int gaus_size;
    int verti_size;
};

class ClfBallVision : public ImgProc {
public:
    ClfBallVision();
    ~ClfBallVision();

public: // 假装是接口的函数
    void imageProcess(cv::Mat input_image, ImgProcResult* output_result);   // 对外接口

    cv::Mat Pretreat(cv::Mat raw_image);                                    // 所有图像进行目标定位前的预处理

    cv::Mat ProcessColor();                                                 // 颜色操作

    std::vector<cv::Rect> GetPossibleRect(cv::Mat binary_image);            // 从二值图获得所有可能区域

    cv::Mat GetHogVec(cv::Rect roi);                                        // 获得src_img 上 roi 的 HOG 特征向量

public: // 真实的接口函数
    void LoadParameters();                                                  // 从文件加载参数

    void StoreParameters();                                                 // 将参数存到文件

    void set_all_parameters(AllParameters);                                 // 调参时候传入参数

    void WriteImg(cv::Mat src, string folder_name, int num);                // 写图片

public: // 数据成员
    cv::Mat src_image_;
    cv::Mat src_hsv_channels_[3];
    cv::Mat used_channel;
    cv::Mat pretreaded_image_;
    cv::Mat thresholded_image_;
    std::vector<cv::Rect> possible_rects_;
    CvSVM svm_classifier_;
    ClfBallResult final_result_;

    // 阈值化相关成员
    int l_min;
    int l_max;
    int s_min;
    int gaus_size;
    int verti_size;

    // 用于获得多个可能结果时候检验
    bool init_former_rect_;
    cv::Rect former_result_rect_;

    // 存图相关
    int start_file_num_;
    int max_file_num_;

    // SVM model path
    string svm_model_name_;

    // 所有在GetPossibleRect得到的待选Rect
    cv::Rect nearest_rect_;

    // 结果Rect
    cv::Rect result_rect_;

    void GetProposal_v2(std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Rect>& result_roi);
    
    int calAverageGary(const Mat &inImg, int &maxGaryDiff, int &averageGrad_xy);
    void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage);
    void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector);
    void calExtendLBPFeature(const Mat &srcImage, Size cellSize, Mat &extendLBPFeature);

    cv::Mat GetUsedChannel(cv::Mat& src_img, int flag);
};



#endif