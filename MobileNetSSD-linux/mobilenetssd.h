#pragma once

#ifndef __MOBILENETSSD_NCNN_H__
#define __MOBILENETSSD_NCNN_H__


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "mobilenet_ssd_voc_ncnn.id.h"
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};



class MOBILENETSSD {

public:
    MOBILENETSSD(string param_path, string bin_path);

    vector<Object> detect(const cv::Mat &bgr);
	cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);

    ~MOBILENETSSD();

private:

    ncnn::Net net;

    const int target_size = 300;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};

};


#endif _
