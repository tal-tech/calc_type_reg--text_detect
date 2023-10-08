#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "math.h"
#include "stdlib.h"
#include <string>
#include <vector>
#include <stdio.h>
#include <numeric>

using namespace std;
using namespace cv;

// 检测模型

class DetApi
{
public:
	// 检测模型初始化
	// datacfg 模型文件
	// cfgfile 模型文件
	// GPU_ID 显卡 默认为0
	// benchmark_layers 默认为0
	DetApi(char *datacfg,char *cfgfile,char *weightfile, int GPU_ID, int benchmark_layers);
//const string& data_file,const string& model_file,const string& weights_file,int GPU_ID);
	~DetApi();
	// 检测模型调用
	// token 输入图像唯一标识符号
	// inMat 输入原始图像
	// thresh 检测阈值
	// hier_thresh 检测阈值
	vector<string> Det_detector(string token,Mat& inMat,float thresh, float hier_thresh);
private:
	image Mat2Image(const cv::Mat& src_img);
	vector<vector<float>> Get_float_result(vector<string> det_Result);
	vector<string> Det_post_process(vector<string> detResult);


	image **alphabet;
private:
	network net;
	char ** names;
};

