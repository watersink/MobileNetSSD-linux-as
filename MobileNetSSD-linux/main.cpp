#include <stdio.h>

#include "mobilenetssd.h"

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN



int main(int argc, char** argv)
{



    MOBILENETSSD mobilenetssd("../assets/mobilenet_ssd_voc_ncnn.param.bin","../assets/mobilenet_ssd_voc_ncnn.bin");
    string imagepath="../ssd.png";

    cv::Mat bgr = cv::imread(imagepath, 1);
    std::vector<Object> objects=mobilenetssd.detect(bgr);
	cv::Mat out_image =mobilenetssd.draw_objects(bgr, objects );
	cv::imwrite("det.jpg",out_image);

    //cv::imshow("image", image);
    //cv::waitKey(0);



    return 0;
}
