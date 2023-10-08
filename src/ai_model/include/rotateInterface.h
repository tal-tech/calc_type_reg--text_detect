#include <iostream>
#include <fstream>
#include "cls_num_rec.hpp"
#include <chrono>
#include <thread>
#include "ocrNumInterface.h"
#include "dataType.h"
#include <sys/timeb.h>
#include<map>
#pragma once
// using namespace std;
// using namespace facethink;
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <string>
//#include "my_util.h"
//#include "dataType.h"

#pragma once

namespace facethink
{
    class NumRecClassify;
}

class jietiRotateModel
{
public:
    jietiRotateModel();
    virtual ~jietiRotateModel();
/*
* *   *初始化模型类
*     * 参数:
*    *IN model_file --- 旋转模型文件.pt文件
*   * IN config_file --- 配置文件
*   * 返回值：
*   * 201 --- 模型初始化失败
*   * 20000  --- 成功
*   *
*   */
    int Init(const std::string& model_file, const std::string& config_file);
    facethink::NumRecClassify* getModel();
private:
    facethink::NumRecClassify* m_model;
};

// class ocrNumInterface
// {
// public:
// 	ocrNumInterface() ;
// 	~ocrNumInterface();
// 	/*
// 	 * 题号手写识别接口
// 	 * IN img --- 待识别的小图
// 	 * OUT structReg --- 接收识别结果
// 	 * IN regNumModel ---- 识别模型指针
// 	 * IN oriLabel  --- 识别的真实内容(标准结果), 默认传空字符串
// 	 * 返回值：
// 	 *310 --- 输入的图像为空
// 	 *311 --- 识别出错
// 	 *20000 --- 识别成功
// 	 * */
// 	int getRegResult(const std::vector<cv::Mat>& vecImg,std::vector<regResult>& vecRegResult,ocrNumRegModel *regNumModel, const std::vector<std::string>& vec_true_label=std::vector<std::string>());
// private:
//     //  RegNum *regNumModel;
// };
// #endif


class rotateInterface
{
  public:
    rotateInterface();

    // 旋转接口
    // rotate_model 旋转模型
    // img_ori  输入图像，会根据旋转模型角度对原图进行旋转。
    // angelIndex  返回角度  0，-90，-180，-270.
    // 返回值:
    //   102 : 模型为空 
    //   103 : 旋转模型出错
    //  20000： 成功
    int getAngle(jietiRotateModel *rotate_model,cv::Mat &img_ori,int &angelIndex);

};
