#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
using namespace cv;
using namespace std;
class Det_api
{
public:
	Det_api();
	static void Init(const string& data_file,const string& model_file, const string& trained_file, std::map<int, int> modelCounts) ;
	vector<pair<Rect,int>>DetObj(string token,const cv::Mat& img,float thresh,float hier_thresh); 
	~Det_api();

};

