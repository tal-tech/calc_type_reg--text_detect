#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include "ocrNumInterface.h"
#include "det_and_reg.h"
#include "dataType.h"
#include <sys/timeb.h>
#include<map>
#pragma once

using namespace std;
using namespace facethink;

class ocrJietiInterface
{
  public:
    ocrJietiInterface();
    ~ocrJietiInterface();
    // ############ 结构化检测结果，给出识别小图 ##############
    // token 唯一标识符号
    // img_gray 传入原始图像
    // det_result 传入检测结果
    // detect_result 结构化的检测结果
    //imgRegV 需要送识别的识别小图。
    // 返回值 
      // 成功 20000
      // 失败 401
    int detMakePair(string token,const cv::Mat img_gray,const vector<string> det_result,vector<LINE_CAL> &detect_result,vector<cv::Mat> &imgRegV);

    //############# 去手写模块 #############
    // detect_result 结构化检测结果
    // vecRegResult 识别结果
    // needregImg 需要二次识别小图
        // 返回值 
      // 成功 20000
      // 失败 402
    int removeHw(vector<LINE_CAL>&detect_result, const vector<regResult> vecRegResult,vector<cv::Mat> &needregImg);
    
    // #################  整合最后结果模块  ################
    // detect_result 接口话检测结果
    // reg2Result 二次识别小图结果
    // text_interface_vector  最终结构结果
        // 返回值 
      // 成功 20000
      // 失败 403
    int getLastResult(vector<LINE_CAL>&detect_result,const vector<regResult>  reg2Result,  vector<TextInterface> &text_interface_vector);

    
    int convetImg2Color(const cv::Mat& imgIn, cv::Mat& imgColor);

    private:
        reg_jieti_removeHw *jieti_removeApi;

};
