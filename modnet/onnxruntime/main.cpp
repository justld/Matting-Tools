#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "MODNet.h"
#include <string>
#include "time.h"


int main()
{
    std::wstring model_path(L"modnet.onnx");
    std::cout << "infer...." << std::endl;
    MODNet modnet(model_path, 1, { 1, 3, 512, 512 });
    clock_t start{ clock() };
    for (int i{ 0 }; i < 20; ++i) {
        modnet.predict_image("C:\\Users\\langdu\\Pictures\\test1.jpeg", "C:\\Users\\langdu\\Pictures\\matting.png");
    }
    std::cout << "time consume:" << (float(clock() - start) / CLOCKS_PER_SEC) << std::endl;
    //modnet.predict_camera(); //Ê¹ÓÃÉãÏñÍ·
    return 0;
}
