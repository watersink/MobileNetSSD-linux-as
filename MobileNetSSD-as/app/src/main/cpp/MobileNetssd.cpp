
#include "MobileNetssd.h"
#include <android/log.h>
#define TAG "mobilessd"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

MOBILENETSSD::MOBILENETSSD(string param_path, string bin_path) {

    const char *param_path_char = param_path.c_str();
    const char *bin_path_char = bin_path.c_str();


    int ret_param = net.load_param_bin(param_path_char);
    int ret_bin = net.load_model(bin_path_char);


	//std::cout<<"### "<<ret_param<<" "<<ret_bin<<std::endl;

}

MOBILENETSSD::MOBILENETSSD(ncnn::Mat param_path, ncnn::Mat bin_path) {


    int ret_param = net.load_param((const unsigned char *)param_path);
    int ret_bin = net.load_model((const unsigned char *)bin_path);

    LOGD("############### %d  %d", ret_param,ret_bin);
    //std::cout<<"### "<<ret_param<<" "<<ret_bin<<std::endl;

}


ncnn::Mat MOBILENETSSD::detect(ncnn::Mat in) {

    int img_w = in.w;
    int img_h = in.h;

    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);


    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(2);

    ex.input(mobilenet_ssd_voc_ncnn_param_id::BLOB_data, in);

    ncnn::Mat out;
    ex.extract(mobilenet_ssd_voc_ncnn_param_id::BLOB_detection_out, out);




    return out;
}




MOBILENETSSD::~MOBILENETSSD() {

    net.clear();
}
