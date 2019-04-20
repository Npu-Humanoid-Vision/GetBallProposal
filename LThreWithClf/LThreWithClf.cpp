#include "LThreWithClf.h"

ClfBallVision::ClfBallVision() {
    final_result_.valid_    = false;
    start_file_num_         = 0;
    max_file_num_           = 500;
    former_result_rect_ = cv::Rect(-1, -1, -1, -1);
    init_former_rect_ = false;
    this->LoadParameters();
}

ClfBallVision::~ClfBallVision() {}

void ClfBallVision::imageProcess(cv::Mat input_image, ImgProcResult* output_result) {
    std::vector<cv::Rect> pos_rects;
    SHOW_IMAGE(input_image, "living");
    cv::Mat for_show_ = input_image.clone();
    src_image_ = input_image.clone();
    pretreaded_image_ = Pretreat(src_image_);
    // cout<<"wocaonima"<<endl;
    thresholded_image_ = ProcessColor();
    // cout<<"nmb"<<endl;
    possible_rects_ = GetPossibleRect(thresholded_image_);  
    // cout<<"yaya"<<endl;

    // once the bot closed to the ball, do it without classifier
    if (possible_rects_.size() <= 3) {
        for (int i=0; i<possible_rects_.size(); i++) {
            pos_rects.push_back(possible_rects_[i]);
            cv::rectangle(for_show_, possible_rects_[i], cv::Scalar(128, 255, 255), 2);
        }
        goto L1;
    }
    for (size_t i=0; i<possible_rects_.size(); i++) {
        cv::Mat roi = src_image_(possible_rects_[i]).clone();
        cv::Mat hog_vec_in_mat = GetHogVec(possible_rects_[i]);

        int lable = (int)svm_classifier_.predict(hog_vec_in_mat);

        if (lable == POS_LABLE) {
            pos_rects.push_back(possible_rects_[i]);
            cv::rectangle(for_show_, possible_rects_[i], cv::Scalar(0, 255, 0), 2);
        }
        else {
            cv::rectangle(for_show_, possible_rects_[i], cv::Scalar(0, 0, 255), 2);
        }
    }
L1: 
    cout<<"pos rect nums: "<<pos_rects.size()<<endl
        <<"with all retc: "<<possible_rects_.size()<<endl;
    if (pos_rects.size() >= 1) {
        final_result_.valid_ = true;
        int max_area = -1;
        int max_idx = -1;
        for (int i=0; i<pos_rects.size(); i++) {
            if (pos_rects[i].area() > max_area) {
                max_area = pos_rects[i].area();
                max_idx = i;
            }
        }
        final_result_.bound_box_ = pos_rects[max_idx];
        final_result_.center_ = cv::Point(final_result_.bound_box_.x + final_result_.bound_box_.width/2, 
                                    final_result_.bound_box_.y + final_result_.bound_box_.height/2);

        cv::rectangle(for_show_, final_result_.bound_box_, cv::Scalar(0, 255, 255));
        cv::circle(for_show_, final_result_.center_, final_result_.bound_box_.width/2, cv::Scalar(255, 255, 0), 3);
    }
    else {
        final_result_.valid_ = false;
    }
    SHOW_IMAGE(for_show_, "result");

    (*dynamic_cast<ClfBallResult*>(output_result)) = final_result_;

#ifndef ADJUST_PARAMETER
    this->WriteImg(src_image_,"src_img",start_file_num_);
    if (final_result_.valid_) {
        cv::rectangle(for_show_, result_rect_, cv::Scalar(0, 255, 255));
    }
    this->WriteImg(for_show_,"center_img",start_file_num_++);
#endif
}

cv::Mat ClfBallVision::Pretreat(cv::Mat raw_image) {
    // // 先获得各个通道先
    cv::Mat t_hsv;
    cv::cvtColor(raw_image, t_hsv, CV_BGR2HSV);
    cv::split(t_hsv, src_hsv_channels_);
    // used_channel = src_hsv_channels_[0].clone();
    cv::Mat blured_image;
    cv::GaussianBlur(raw_image, blured_image, cv::Size(2*gaus_size+1, 2*gaus_size+1), 0, 0);

    used_channel = GetUsedChannel(blured_image, L);
    SHOW_IMAGE(blured_image, "gaused image");
    return blured_image;
}

cv::Mat ClfBallVision::ProcessColor() {
    cv::Mat mask = src_hsv_channels_[1] >= s_min;
    cv::Mat thre_result;

    thre_result = used_channel >= l_min & used_channel <= l_max;
    thre_result = thre_result & mask;

    // for grass 
    cv::Mat glass_mask = cv::Mat(this->src_image_.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat t_a = GetUsedChannel(this->pretreaded_image_, A);
    cv::Mat glass_thre = t_a >= a_min & t_a <= a_max;
    
    int last_i = -1;
    int last_j = -1;
    for (int i = 0; i < glass_thre.cols; i++) {
        for (int j = 0; j < glass_thre.rows-3; j++) {
            if (glass_thre.at<uchar>(j, i) == 255
                && glass_thre.at<uchar>(j+1, i) == 255
                && glass_thre.at<uchar>(j+2, i) == 255) {
                if (last_i < 0) {
                    last_i = i;
                    last_j = j;
                } else {
                    cv::line(glass_mask, cv::Point(i, j), cv::Point(last_i, last_j), cv::Scalar(255), 3);
                    last_i = i;
                    last_j = j;
                }
                break;
            }
        }
    }
    // cv::line(glass_mask, cv::Point(last_i, last_j), cv::Point(src_image_.rows, last_j), cv::Scalar(255), 3);
    // SHOW_IMAGE(glass_mask, "grass edge");
    cv::floodFill(glass_mask, cv::Point(glass_thre.cols / 2, glass_thre.rows - 1), 255);
    SHOW_IMAGE(glass_mask, "grass mask");
    thre_result = thre_result & glass_mask;

    // cout<<hori_size<<endl;
    cv::Mat hori_kernal = cv::getStructuringElement(MORPH_RECT, Size(hori_size, 1));
    // cout<<"yayaya"<<endl;
    cv::Mat veri_kernal = cv::getStructuringElement(MORPH_RECT, Size(1, verti_size));

    // cv::Mat dilate_kernal = cv::getStructuringElement(MORPH_RECT, Size(3, 3));
    cv::dilate(thre_result, thre_result, hori_kernal);
    cv::erode(thre_result, thre_result, veri_kernal);
    // cv::dilate(thre_result, thre_result, dilate_kernal);
    SHOW_IMAGE(thre_result, "l_thre");
    return thre_result;
}

std::vector<cv::Rect> ClfBallVision::GetPossibleRect(cv::Mat binary_image) {
    std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > contours_poly;
    std::vector<cv::Rect> bound_rect;

    cv::Mat image_for_contours = binary_image.clone();
    cv::findContours(image_for_contours, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    contours_poly.resize(contours.size());
    // bound_rect.resize();

    // double max_area = 0.0;
    // int max_area_idx = -1;
    int max_inter_area = 0.0;
    int min_dist_idx = -1;
    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, false);
        cv::Rect t_rect = cv::boundingRect(contours_poly[i]);
        double wh_rate = t_rect.width*1.0/t_rect.height;
 
        // shape & area thre
        // if (0.33<wh_rate && wh_rate<3.0 && t_rect.area() > 100) {
        if (t_rect.width*t_rect.width > 100) {
            if (t_rect.width < t_rect.height) {
                continue;
            }
            else if (t_rect.width > t_rect.height) {

                // middle rect without debug
                int max_sum = -1;
                int max_x = -1;
                for (int i=t_rect.x; i<t_rect.x+t_rect.width; i++) {
                    int pix_sum = 0;
                    for (int j=t_rect.y; j<t_rect.y+t_rect.height; j++) {
                        if (binary_image.at<uchar>(j, i) == 255) {
                            pix_sum += 1;
                        }
                    }
                    if (pix_sum > max_sum) {
                        max_sum = pix_sum;
                        max_x = i;
                    }
                }
                // get the middle rect
                if (max_x-t_rect.height/2 > 0 
                    && max_x-t_rect.height/2+t_rect.height*3/2 < src_image_.cols
                    && t_rect.y+t_rect.height*3/2 < src_image_.rows ) {
                    cv::Rect middle_rect;
                    middle_rect = cv::Rect(max_x-t_rect.height/2, t_rect.y, t_rect.height*3/2, t_rect.height*3/2);
                        // middle_rect = cv::Rect(max_x-t_rect.height/2, t_rect.y, t_rect.height, t_rect.height);
                    
                    int delta = verti_size;
                    if (middle_rect.x - delta > 0) {
                        middle_rect.x -= delta;
                    }
                        
                    if (middle_rect.y - delta > 0) {
                        middle_rect.y -= delta;
                    }

                    if (middle_rect.width+middle_rect.x+delta*2 < src_image_.cols)
                        middle_rect.width += delta*2;
                    if (middle_rect.height+middle_rect.y+delta*2 < src_image_.rows)
                        middle_rect.height += delta*2;

                    // bound_rect.push_back(middle_rect);
                }
                

                /// abandon version
                // int step = t_rect.height/5;
                // for (int i=t_rect.x; i+t_rect.height<t_rect.x+t_rect.width; i+=step) {
                //     int delta = t_rect.height/2;
                //     cv::Rect tt_rect = cv::Rect(i, t_rect.y, t_rect.height, t_rect.height);
                //     if (tt_rect.x-delta > 0) 
                //         tt_rect.x -= delta;
                //     if (tt_rect.y-delta >0)
                //         tt_rect.y -= delta;

                //     if (tt_rect.width+tt_rect.x+delta < src_image_.cols) {
                //         tt_rect.width += delta;
                //         // cout<<1<<endl;
                //     }
                        
                //     if (tt_rect.height+tt_rect.y+delta < src_image_.rows) {
                //         tt_rect.height += delta;
                //         // cout<<2<<endl;
                //     }
                        

                //     bound_rect.push_back(tt_rect);
                // }

                int delta = this->verti_size;

                if (t_rect.x-delta > 0) 
                    t_rect.x -= delta;
                if (t_rect.y-delta >0)
                    t_rect.y -= delta;

                if (t_rect.y + t_rect.width < src_image_.rows)
                    t_rect.height = t_rect.width;   
                if (t_rect.width+t_rect.x+delta*2 < src_image_.cols)
                    t_rect.width += delta*2;
                if (t_rect.height+t_rect.y+delta*2 < src_image_.rows)
                    t_rect.height += delta*2;
                // cout<<t_rect<<endl;
                bound_rect.push_back(t_rect);
            }
            else {
                // cout<<t_rect<<endl;
                bound_rect.push_back(t_rect);
            }

        }

        // bound_rect[i] = cv::boundingRect(contours_poly[i]);
        
        // if (cv::contourArea(contours_poly[i], false) > max_area) {
        //     max_area = cv::contourArea(contours_poly[i], false);
        //     max_area_idx = i;
        // }

    }
    return bound_rect;
}

cv::Mat ClfBallVision::GetHogVec(cv::Rect roi) {
    cv::Mat roi_in_mat = src_image_(roi).clone();
    // SHOW_IMAGE(roi_in_mat);
    cv::resize(roi_in_mat, roi_in_mat, cv::Size(128, 128)); // 与训练相关参数，之后最好做成文件传入参数
    // SHOW_IMAGE(roi_in_mat);
    cv::HOGDescriptor hog_des(Size(128, 128), Size(16,16), Size(8,8), Size(8,8), 9);
    std::vector<float> hog_vec;
    hog_des.compute(roi_in_mat, hog_vec);

    for (int j=0; j<6; j++) {
        cv::Mat ROI_l = GetUsedChannel(roi_in_mat, j);
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
    // cout<<t<<endl;
    cv::Mat hog_vec_in_mat = t.t();
    // cout<<hog_vec_in_mat<<endl;
    hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);
    return hog_vec_in_mat;
}



void ClfBallVision::LoadParameters() {
#ifdef ADJUST_PARAMETER
    std::ifstream in_file("./7.txt");

#else    
    std::ifstream in_file("../source/data/set_sprint_param/7.txt");
#endif
    if (!in_file) {
        cerr<<"Error:"<<__FILE__
                <<":line"<<__LINE__<<endl
                <<"     Complied on"<<__DATE__
                <<"at"<<__TIME__<<endl;
    }
    int i = 0;
    string line_words;
    cout<<"Loading Parameters"<<endl;
    while (in_file >> line_words) {
        cout<<line_words<<endl;
        std::istringstream ins(line_words);
        switch (i++) {
        case 0:
            ins >> l_min;
            break;
        case 1:
            ins >> l_max;
            break;
        case 2:
            ins >> a_min;
            break;
        case 3:
            ins >> a_max;
            break;
        case 4:
            ins >> s_min;
            break;
        case 5:
            ins >> gaus_size;
            break;
        case 6:
            ins >> verti_size;
            break;
        case 7:
            ins >> hori_size;
            break;
        case 8:
            ins >> svm_model_name_;
            break;
        }
    }
#ifdef ADJUST_PARAMETER
    svm_classifier_.load(svm_model_name_.c_str());
#else
    svm_classifier_.load(("../source/data/set_sprint_param/"+svm_model_name_).c_str());
#endif
}

void ClfBallVision::StoreParameters() {
    std::ofstream out_file("./7.txt");
    if (!out_file) {
        cerr<<"Error:"<<__FILE__
                <<":line"<<__LINE__<<endl
                <<"     Complied on"<<__DATE__
                <<"at"<<__TIME__<<endl;
    }
    out_file << setw(3) << setfill('0') << l_min                        <<"___l_min"<<endl;
    out_file << setw(3) << setfill('0') << l_max                        <<"___l_max"<<endl;
    out_file << setw(3) << setfill('0') << a_min                        <<"___a_min"<<endl;
    out_file << setw(3) << setfill('0') << a_max                        <<"___a_max"<<endl;
    out_file << setw(3) << setfill('0') << s_min                        <<"___s_min"<<endl;
    out_file << setw(3) << setfill('0') << gaus_size                    <<"___gaus_size"<<endl;
    out_file << setw(3) << setfill('0') << verti_size                   <<"___verti_size"<<endl;
    out_file << setw(3) << setfill('0') << hori_size                    <<"___hori_size"<<endl;
    out_file << svm_model_name_;
    out_file.close();
}

void ClfBallVision::set_all_parameters(AllParameters ap) {
    l_min = ap.l_min;
    l_max = ap.l_max;
    a_min = ap.a_min;
    a_max = ap.a_max;
    s_min = ap.s_min;
    gaus_size = ap.gaus_size;
    verti_size = ap.verti_size;
    hori_size = ap.hori_size;
}
     
void ClfBallVision::WriteImg(cv::Mat src, string folder_name, int num) {
    stringstream t_ss;
    string path = "../source/data/con_img/";
    if (start_file_num_ <= max_file_num_) {
        path += folder_name;
        path += "/";

        t_ss << num;
        path += t_ss.str();
        t_ss.str("");
        t_ss.clear();
        // path += std::to_string(num); 

        path += ".jpg";

        cv::imwrite(path,src);
    }
}



 
//计算输入图片的最大灰度差、平均灰度、平均梯度
int ClfBallVision::calAverageGary(const Mat &inImg, int &maxGaryDiff, int &averageGrad_xy)
{
	float averageGary;
	int garySum = 0;
	int i, j;
 
	//求平均灰度值
	for (i=0; i<inImg.cols; i++)
	{
		for (j=0; j<inImg.rows; j++)
		{
			garySum += inImg.at<uchar>(j, i);
		}
	}
	averageGary = (int)(garySum*1.0f/(inImg.rows*inImg.cols));
 
	//求滑窗内的最大灰度差值
	double minGary, maxGary; 
	minMaxLoc(inImg, &minGary, &maxGary, NULL, NULL);
	maxGaryDiff = (int)(maxGary-minGary);
 
	//求滑窗内的平均梯度值
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad_xy; 
	Sobel( inImg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT );  	//求X方向梯度 
	convertScaleAbs( grad_x, abs_grad_x );  
	Sobel( inImg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT );  	//求Y方向梯度  
	convertScaleAbs( grad_y, abs_grad_y );  
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_xy);  	       //合并梯度(近似)  
	//cout<<"gary_xy"<<grad_xy<<endl;
 
	int grad_xy_sum = 0;
	for (i=0; i<inImg.cols; i++)
	{
		for (j=0; j<inImg.rows; j++)
		{
			grad_xy_sum += grad_xy.at<uchar>(j, i);
		}
	}
	averageGrad_xy = (int)(grad_xy_sum*1.0f/(inImg.rows*inImg.cols));
	return averageGary;
}
 
 
// 计算等价模式LBP特征图
void ClfBallVision::ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage)
{
	// 参数检查，内存分配
	CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
	LBPImage.create(srcImage.size(), srcImage.type());
 
	// 计算LBP图
	Mat extendedImage;
	copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, BORDER_DEFAULT);
 
	// LUT(256种每一种模式对应的等价模式)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };
 
	// 计算LBP
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBP=LBPImage.cols;
	uchar *rowOfExtendedImage = extendedImage.data+widthOfExtendedImage+1;
	uchar *rowOfLBPImage = LBPImage.data;
 
	int pixelDiff = 5;
 
	for (int y = 1; y <= heightOfExtendedImage - 2; ++y,rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBPImage = rowOfLBPImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
		{
			// 计算LBP值
			int LBPValue = 0;
			if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 128;
			if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 64;
			if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 32;
			if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 16;
			if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 8;
			if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 4;
			if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 2;
			if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 1;
 
			colOfLBPImage[0] = table[LBPValue];
		}
	}
}
 
//计算归一化的LBP特征矩阵
void ClfBallVision::ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	// 参数检查，内存分配
	CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
 
	Mat LBPImage;
	ComputeLBPImage_Uniform(srcImage, LBPImage);
 
	//cout<<"LBPImage_uniform："<<endl<<LBPImage<<endl<<endl;
 
	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;
 
	// 特征向量的个数
	int numberOfDimension = 58 * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));
 
	// 计算LBP特征向量
	int stepOfCell=srcImage.cols;
	int index = -58;// cell的特征向量在最终特征向量中的起始位置
	float *dataOfFeatureVector=(float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index+=58;
 
			// 计算每个cell的LBP直方图
			Mat cell = LBPImage(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			uchar *rowOfCell=cell.data;
			int sum = 0; // 每个cell的等价模式总数
			for(int y_Cell=0;y_Cell<=cell.rows-1;++y_Cell,rowOfCell+=stepOfCell)
			{
				uchar *colOfCell=rowOfCell;
				for(int x_Cell=0;x_Cell<=cell.cols-1;++x_Cell,++colOfCell)
				{
					if(colOfCell[0]!=0)
					{
						// 在直方图中转化为0~57，所以是colOfCell[0] - 1
						++dataOfFeatureVector[index + colOfCell[0]-1];
						++sum;
					}
				}
			}
 
			for (int i = 0; i <= 57; ++i)
				dataOfFeatureVector[index + i] /= sum;
		}
	}
}
 
//计算扩展LBP特征矩阵，在原LBP特征上增加了3维
void ClfBallVision::calExtendLBPFeature(const Mat &srcImage, Size cellSize, Mat &extendLBPFeature)
{
	// 参数检查，内存分配
	CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
 
	Mat LBPImage;
	int i,j, height, width;
	height = srcImage.rows;
	width = srcImage.cols;
 
	ComputeLBPFeatureVector_Uniform(srcImage, cellSize, LBPImage);   //求归一化后的LBP特征
	//cout<<"LBPImage"<<LBPImage<<endl;   //取值范围[0,58]
 
	//把LBPImage折算到[0,255]之间
	Mat LBPImage_255(1, LBPImage.cols, CV_8UC1, Scalar(0));
	for (i=0; i<LBPImage.cols; i++)
	{
		LBPImage_255.at<uchar>(0,i) = (uchar)(LBPImage.at<float>(0,i) * 255.0f);
	}
	//cout<<"LBPImage_255"<<endl<<LBPImage_255<<endl;
 
	int maxGaryDiff, averageGrad_xy;
	int averageGary = calAverageGary(srcImage, maxGaryDiff, averageGrad_xy);
	//cout<<"averageGary="<<averageGary<<",   maxGrayDiff="<<maxGaryDiff<<endl<<endl;
 
	int descriptorDim;
	descriptorDim = LBPImage.cols + 3;
	Mat extendLBPFeature_255 = Mat::zeros(1, descriptorDim, CV_8UC1); 
 
	for (i=0; i<LBPImage.cols; i++)
	{
		extendLBPFeature_255.at<uchar>(0,i) = LBPImage_255.at<uchar>(0,i);
	}
	extendLBPFeature_255.at<uchar>(0,LBPImage.cols) = averageGary;       //增加维度，存放平均像素
	extendLBPFeature_255.at<uchar>(0,LBPImage.cols+1) = maxGaryDiff;     //增加维度，存放最大灰度差
	extendLBPFeature_255.at<uchar>(0,LBPImage.cols+2) = averageGrad_xy;  //增加维度，存放平均梯度
 
	//把扩展LBP特征矩阵归一化
	extendLBPFeature = Mat(1, descriptorDim, CV_32FC1, Scalar(0)); 
	for(i=0; i<descriptorDim; i++)
	{
		extendLBPFeature.at<float>(0,i) = extendLBPFeature_255.at<uchar>(0,i)*1.0f/255;
	}
	//cout<<"extendLBPFeature： "<<endl<<extendLBPFeature<<endl;
}

void ClfBallVision::GetProposal_v2(std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Rect>& result_roi) {
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

                if (t_rect.y + t_rect.width < src_image_.rows)
                    t_rect.height = t_rect.width;   
                if (t_rect.width+t_rect.x+delta < src_image_.cols)
                    t_rect.width += delta;
                if (t_rect.height+t_rect.y+delta < src_image_.rows)
                    t_rect.height += delta;
                // cout<<t_rect<<endl;
                result_roi.push_back(t_rect);
            }
            else {
                // cout<<t_rect<<endl;
                result_roi.push_back(t_rect);
            }

        }
    }
}


cv::Mat ClfBallVision::GetUsedChannel(cv::Mat& src_img, int flag) {
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
