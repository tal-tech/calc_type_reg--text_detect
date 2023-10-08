#include <iostream>
#include "ocrNumInterface.h"
#include <vector>
using namespace std;
#pragma once

// 识别结果
typedef struct tagRegResult
{
	std::vector<std::string> vecCondidateResult; //识别候选结果
	std::vector<std::vector<int> > vecCondidateResultPos; //所有候选结果的每个字符的位置，需要对应到识别输入的原图
	std::vector<float> vecProb;//每个候选结果的识别置信度
	int nType; //类型 0---题号，  1--- 英语手写
}regResult;


// 普通式子行结构体（中间变量）
struct LINE_CAL
{
    std::vector<int> rect_line;
    cv::Rect rect_line_rect;
    cv::Rect rect_hw;
    cv::Rect rect_hw_bigcirc;
    cv::Mat rect_img;
    cv::Mat rect_nhw_img;
    
    float iou_score;
    int hw_flag,hw_right;
    int hw_bigcirc_flag = 1;
    int bigcirc_flag;
    string reg_result;
    string remove_hw_result;
    string last_result;
    float prob;
};

// 竖式结构体
struct VER
{
    std::vector<int> rect_line;
    cv::Rect rect_line_rect;
    std::string  mulLine_result;
    float  prob;
    vector<LINE_CAL> verLineVector;
    string ver_reg;
};

// 最终返回结构体
struct TextInterface
{
    // 两个坐标位置。
    std::vector<int> rect_line;
    cv::Rect rect_hw;

    
    cv::Rect rect_line_rect;  //最终位置，x,y,w,h
    std::string text;  // 最终的text
    float prob;
};








