//#include "calc_type_reg.h"
#include <vector>

#include <string>
#include <mutex>
#include "DetApi.h"
#include "cls_num_rec.hpp"
#include "rotateInterface.h"
#include "ocrJietiInterface.h"
#include"opencv2/opencv.hpp"
#include"json/json.h"
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>
std::mutex s_mutex_reg;
std::mutex s_mutex_rotate;
std::mutex s_mutex_detect;

DetApi* s_detect_model = nullptr;
ocrNumRegModel* s_reg_pt_mix = nullptr;
jietiRotateModel* s_rotate_model = nullptr;
cv::Mat cv_image_;

void InitAiModel(){
    //################### 初始化旋转模型 ################
    s_rotate_model = new jietiRotateModel();
    int init_flag = s_rotate_model->Init("./ai_model/model/rotate_model/rot_0922.pt","./ai_model/model/config.ini");
    std::cout<< "rotate init result: " << std::endl;

    //###############初始化检测模型#########################
    s_detect_model = new DetApi("./ai_model/model/model_det/obj.data","./ai_model/model/model_det/yolov4_512.cfg","./ai_model/model/model_det/yolov4_512_0922.weights",0, 0);
    std::cout<< "detect_model init success!"<<endl;

    //############### 初始化识别模型 #########################
    s_reg_pt_mix = new ocrNumRegModel();
    init_flag = s_reg_pt_mix->Init("./ai_model/model/crnn_0907.pt", "./ai_model/model/config.ini");
    std::cout << "reg_pt_mix model init result: " << init_flag<<std::endl;
}

void ReleaseAiModel(){
    if(s_detect_model)
        delete s_detect_model;
    s_detect_model = nullptr;

    if(s_reg_pt_mix)
        delete s_reg_pt_mix;
    s_reg_pt_mix = nullptr;

    if(s_rotate_model)
        delete s_rotate_model;
    s_rotate_model = nullptr;
}


bool handler(Json::Value &data_result) {

    bool err_code = true;

    do{
        cv::Mat img_dest;
        ocrJietiInterface ocrjietiModel;
        if(ocrjietiModel.convetImg2Color(cv_image_, img_dest) != 0){
            err_code = false;
           std::cout << "convert image type failed, reuestId: " << std::endl;
            break;

        }
        // ############## 旋转  #############
        {
            std::lock_guard<std::mutex> lock(s_mutex_rotate);
            int rotate_index = 0;
            rotateInterface rotate_inference;
            int res = rotate_inference.getAngle(s_rotate_model, img_dest, rotate_index);
            if(res != 20000){
                
                std::cout << "algorithm rotate failed: " << res << ", reuestId: " << std::endl;
                break;
            }
            data_result["rotate"] = rotate_index;
        }

        boost::uuids::uuid a_uuid = boost::uuids::random_generator()();
        std::string image_token = boost::uuids::to_string(a_uuid);

        //####### 检测对象 ##############
        vector<std::string> check_result;
        cv::Mat imgdet = img_dest.clone();
        {
            std::lock_guard<std::mutex> lock(s_mutex_detect);
            check_result = s_detect_model->Det_detector(image_token, imgdet,0.25,0.5);
            // data_result["check_result"] =check_result
            std::cout<<"11111111111111111111111111"<<std::endl;
            for (const auto &item : check_result) {
                std::cout << item << ' ';
            }
        }

        

        //####### 识别对象 ##############
        vector<cv::Mat> imgRegV;
        vector<LINE_CAL> detect_result;

        //接口模块，去手写模块
        int res = ocrjietiModel.detMakePair(image_token,img_dest,check_result,detect_result,imgRegV);
        if(res != 20000){
            err_code = false;
            std::cout << "algorithm detect make pair failed: " << res << ", reuestId: " << std::endl;
            break;
        }

        int img_num = imgRegV.size();
        int batch_size = 5;
        int group_num = ceil(img_num/(double)batch_size);

        ocrNumInterface classify;
        std::vector<regResult> vecRegResult, vec2regResult, vecBatchResult;

        for (int cur_index = 0; cur_index < group_num; cur_index++)
        {
            vecBatchResult.clear();
            int begin_index = cur_index * batch_size;
            int end_index = min((cur_index+1)*batch_size, img_num);
            vector<cv::Mat> imgBatch(imgRegV.begin() + begin_index,imgRegV.begin() + end_index);

            {
                std::vector<string> vecOriLabel;
                std::lock_guard<std::mutex> lock(s_mutex_reg);
                if (classify.getRegResult(imgBatch,vecBatchResult, s_reg_pt_mix, vecOriLabel) != 20000){
                    continue;
                }
            }

            vecRegResult.insert(vecRegResult.end(), vecBatchResult.begin(), vecBatchResult.end());
        }

        vector<cv::Mat> reg2ImgV;
        res = ocrjietiModel.removeHw(detect_result, vecRegResult,reg2ImgV);
        if(res != 20000){
            err_code = false;
            std::cout << "algorithm remove handwriting failed: " << res << ", reuestId: " << std::endl;
            break;
        }

        {
            std::vector<string> label2;
            std::lock_guard<std::mutex> lock(s_mutex_reg);
            res = classify.getRegResult(reg2ImgV, vec2regResult, s_reg_pt_mix, label2);
            if(res != 20000 && res != 301){
                err_code = false;
                std::cout << "recognition algorithm failed: " << res << ", reuestId: " << std::endl;
                break;
            }
        }

        std::vector<TextInterface> text_interface_vector;
        res = ocrjietiModel.getLastResult(detect_result,vec2regResult,text_interface_vector);
        if(res != 20000){
            err_code = false;
            std::cout << "algorithm Integrate final results failed: " << res << ", reuestId: " << std::endl;
            break;
        }

        data_result["result"] = Json::arrayValue;
        for(auto text_result : text_interface_vector){
            Json::Value json_result, json_position;
            json_position["x"] = int(text_result.rect_line_rect.x);
            json_position["y"] = int(text_result.rect_line_rect.y);
            json_position["width"] = int(text_result.rect_line_rect.width);
            json_position["height"] = int(text_result.rect_line_rect.height);

            json_result["text"] = text_result.text;
            json_result["position"] = json_position;
            json_result["confidence"] = text_result.prob;

            data_result["result"].append(json_result);
        }
    }while(false);

    return err_code;
}


int main(){
    cv_image_=cv::imread("./test.png");
    InitAiModel();
    Json::Value res;
    handler(res);
    // std::cout<<res<<std::endl;
    Json::Value results = res["result"];
    for (int i = 0; i < results.size(); ++i) {
        std::string text = results[i]["text"].asString();
        std::cout << "Text for result " << i+1 << " is: ";
        std::cout << text << std::endl;
    }
    ReleaseAiModel();

    return 0;


}
