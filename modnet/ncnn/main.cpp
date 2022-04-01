#include <opencv.hpp>
#include <iostream>
#include "MODNet.h"
#include <vector>
#include "net.h"
#include "time.h"


int main() {
	//std::string param_path{ "onnx_model\\simple_modnet.param" };
	//std::string bin_path{ "onnx_model\\simple_modnet.bin" };
	std::string param_path{ "onnx_model\\modnet_int8.param" };
	std::string bin_path{ "onnx_model\\modnet_int8.bin" };
	std::vector<int> input_shape{ 1, 3, 512, 512 };
	MODNet model(param_path, bin_path, input_shape);

	model.predict_image("C:\\Users\\xx\\Pictures\\test1.jpeg", "C:\\Users\\xx\\Pictures\\predict.png");

	// ‘§≤‚≤¢œ‘ æ
	//cv::Mat image = cv::imread("C:\\Users\\langdu\\Pictures\\test.png");
	//cv::Mat segFrame = model.predict_image(image);
	//cv::imshow("1", segFrame);
	//cv::waitKey(0);

	// …„œÒÕ∑
	//model.predict_camera();
	return -1;
}

