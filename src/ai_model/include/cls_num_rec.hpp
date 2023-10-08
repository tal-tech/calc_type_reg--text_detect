///////////////////////////////////////////////////////////////////////////////////////
///  Copyright (C) 2017, TAL AILab Corporation, all rights reserved.
///
///  @file: cls_num_rec.hpp
///  @brief 数字公式识别
///  @details 最初版本
//
//
///  @version 1.0.0.0
///  @author Qiao Yu
///  @date 2020-05-07
///
///  @see 使用参考：demo.cpp
///
///////////////////////////////////////////////////////////////////////////////////////
#ifndef __FACETHINK_API_CLS_NUM_REC_HPP__
#define __FACETHINK_API_CLS_NUM_REC_HPP__
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>
#ifdef WIN32
#ifdef DLL_EXPORTS
#define EXPORT_CLASS   __declspec(dllexport)
#define EXPORT_API  extern "C" __declspec(dllexport)
#define EXPORT_CLASS_API
#else
#define EXPORT_CLASS   __declspec(dllimport )
#define EXPORT_API  extern "C" __declspec(dllimport )
#endif
#else
#define EXPORT_CLASS
#define EXPORT_API  extern "C" __attribute__((visibility("default")))
#define EXPORT_CLASS_API __attribute__((visibility("default")))
#endif
namespace facethink {
	class EXPORT_CLASS NumRecClassify {
	public:
		EXPORT_CLASS_API explicit NumRecClassify(void);
		EXPORT_CLASS_API virtual ~NumRecClassify(void);

		//初始化
		//det_model_file是模型文件
		//config_file为配置文件
		EXPORT_CLASS_API static NumRecClassify* create(	const std::string& det_model_file,const std::string& config_file);
		EXPORT_CLASS_API static NumRecClassify* create_rotate(	const std::string& det_model_file,const std::string& config_file);

		//输入batch img
		//输出crnn模型的输出
		//返回值-1：输入vector为空
		//返回值-2：输入channel不为1
		EXPORT_CLASS_API virtual int detection(std::vector<cv::Mat>& imgs,  std::vector<std::vector<std::vector<float>>>& probs) = 0;
		EXPORT_CLASS_API virtual int detection(std::vector<cv::Mat>& imgs,  std::vector<std::vector<float>>& probs) = 0;
	};
}
#endif
