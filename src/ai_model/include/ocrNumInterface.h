
#ifndef OCRNUMINTERFACE_H
#define OCRNUMINTERFACE_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
// #include "my_util.h"
#include "dataType.h"

namespace facethink
{
    class NumRecClassify;
}

class ocrNumRegModel
{
public:
    ocrNumRegModel();
    virtual ~ocrNumRegModel();
/*
* *   *初始化模型类
*     * 参数:
*    *IN model_file --- 模型文件.pt文件
*   * IN config_file --- 配置文件
*   * 返回值：
*   * 300 --- 模型初始化失败
*   * 20000  --- 成功
*   *
*   */
    int Init(const std::string& model_file, const std::string& config_file);
    facethink::NumRecClassify* getModel();
private:
    facethink::NumRecClassify* m_model;
};

class ocrNumInterface
{
public:
	ocrNumInterface() ;
	~ocrNumInterface();
	/*
	 * 题号手写识别接口
	 * IN img --- 待识别的小图
	 * OUT structReg --- 接收识别结果
	 * IN regNumModel ---- 识别模型指针
	 * IN oriLabel  --- 识别的真实内容(标准结果), 默认传空字符串
	 * 返回值：
	 *301 --- 输入的图像为空
	 *302 --- 识别结果有问题
	 *303 --- 识别出错

	 *20000 --- 识别成功
	 * */
	int getRegResult(const std::vector<cv::Mat>& vecImg,std::vector<regResult>& vecRegResult,ocrNumRegModel *regNumModel, const std::vector<std::string>& vec_true_label=std::vector<std::string>());
private:
    //  RegNum *regNumModel;
};
#endif
