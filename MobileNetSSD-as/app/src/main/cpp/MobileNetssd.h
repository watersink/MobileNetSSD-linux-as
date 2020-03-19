//#pragma once

#ifndef __MOBILENETSSD_NCNN_H__
#define __MOBILENETSSD_NCNN_H__


#include "net.h"
#include "mobilenet_ssd_voc_ncnn.id.h"
#include <iostream>
#include <vector>
using namespace std;



class MOBILENETSSD {

public:
    MOBILENETSSD(string param_path, string bin_path);
	MOBILENETSSD(ncnn::Mat param_path, ncnn::Mat bin_path);

	ncnn::Mat detect(ncnn::Mat in);

    ~MOBILENETSSD();

private:

    ncnn::Net net;

    const int target_size = 300;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};

};


#endif
