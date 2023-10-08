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

using namespace std;
using namespace cv;
class DetApi
{
public:
	DetApi(char *datacfg,char *cfgfile,char *weightfile, int GPU_ID, int benchmark_layers);
//const string& data_file,const string& model_file,const string& weights_file,int GPU_ID);
	~DetApi();
	vector<string> Det_detector(string token,Mat& inMat,float thresh, float hier_thresh);
private:
	image Mat2Image(const cv::Mat& src_img);

	image **alphabet;
private:
	network net;
	char ** names;
};

