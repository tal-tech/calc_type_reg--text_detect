#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include "ocrNumInterface.h"
#include "dataType.h"
#include <sys/timeb.h>
#include<map>
#pragma once

using namespace std;
using namespace facethink;

class reg_jieti_removeHw
{
  public:
    reg_jieti_removeHw();
    //######old###################
    int getResult(ocrNumRegModel* regModel,string detoneName,cv::Mat imgMat);
    //######new###################
    void getLastResult(vector<LINE_CAL>&detect_result,const vector<regResult>  reg2Result,  vector<TextInterface> &text_interface_vector);
    void removeHw(vector<LINE_CAL>&detect_result,const vector<regResult> vecRegResult,vector<cv::Mat> &needregImg);
    void detResultSplit(const vector<string> det_result);
    void detMakePair(string token,const cv::Mat img_gray,const vector<string> det_result,vector<LINE_CAL> &detect_result,vector<cv::Mat> &imgRegV);
    void getdetResult(string detoneName,vector<vector<int>> &det_posV);

  private:   
    cv::Mat img_ori;   // 原始图像img_W
    cv::Mat img_gray;   // 原始灰度
    int img_H;
    int img_W;
    std::string token;
    vector<int> reg2Flag;
    vector<vector<int>> rect_normal;  //对应0颜色
    vector<vector<int>> rect_ver;   //对用1颜色
    vector<vector<int>> rect_verline; //对应2颜色
    vector<vector<int>> rect_hw;     //对应3颜色
    vector<vector<int>> rect_hw_bigcirc; //对用4颜色
    vector<vector<int>> ver_img_map;
    vector<VER> ver_vector;

    int getmilliontime();
    std::vector<std::string> split(std::string str, std::string pattern);

    void clearVar(vector<LINE_CAL> &detect_result_pair);
    float cal_iou(vector<int> a, vector<int> b,int & hw_right_flag);
    int rect_pos_stand(vector<int> &rect_det,const int rect_index);
    void makeLineCalstruct(const vector<int> lineOne, LINE_CAL &rect_calculate);
    void makeVerPair(vector<VER> &verVector,vector<cv::Mat> &imgAllMat);
    void removeVerHw(vector<VER>&ver_vector,const vector<regResult> vecRegResult,vector<cv::Mat> &needregImg);
    void reg_add_bigcirc(string &input_latex);
    void normalReplace(string& str, const string& old_value, const string& new_value);
    void plot_result(const cv::Mat imgMat,vector<TextInterface> detect_result_pair,string out_path,string result_f_path);

    map<int, cv::Scalar>  m_structColor= {
      map<int, cv::Scalar>::value_type(0, cv::Scalar(0, 0, 255)),		  //式子行框红色
      map<int, cv::Scalar>::value_type(1, cv::Scalar(0, 255, 255)),	  //表格黄色;	
			map<int, cv::Scalar>::value_type(2, cv::Scalar(255, 0, 0)),		  //式子行框蓝色
			map<int, cv::Scalar>::value_type(3, cv::Scalar(255, 255, 0)),	  //浅蓝色
			map<int, cv::Scalar>::value_type(4, cv::Scalar(255, 155, 48)),	 //手写公式紫色
			map<int, cv::Scalar>::value_type(5, cv::Scalar(0, 255, 0)),	  //印刷式子绿色
			map<int, cv::Scalar>::value_type(6, cv::Scalar(0, 165, 255)),	  //图形橘色   
    };
};
