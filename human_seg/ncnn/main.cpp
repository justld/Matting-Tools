#include <opencv.hpp>
#include <iostream>
#include "HumanSeg.h"
#include <vector>
#include "net.h"


int main() {
	std::string param_path{ "D:\\C_code\\humanseg_ncnn\\onnx_model\\simple_model_interp.param" };
	std::string bin_path{ "D:\\C_code\\humanseg_ncnn\\onnx_model\\simple_model_interp.bin" };
	std::vector<int> input_shape{ 1, 3, 192, 192 };
	HumanSeg model(param_path,bin_path,input_shape);
	// Ԥ�Ⲣ����
	//model.predict_image("C:\\Users\\langdu\\Pictures\\test1.jpeg", "C:\\Users\\langdu\\Pictures\\predict.png");
	 
	// Ԥ�Ⲣ��ʾ
	//cv::Mat image = cv::imread("C:\\Users\\langdu\\Pictures\\test.png");
	//cv::Mat segFrame = model.predict_image(image);
	//cv::imshow("1", segFrame);
	//cv::waitKey(0);

	// ����ͷ
	model.predict_camera();
	return -1;
}

