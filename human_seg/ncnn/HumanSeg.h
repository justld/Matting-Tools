#pragma once
#include <string>
#include "net.h"
#include <opencv.hpp>
#include "time.h"


class HumanSeg
{
private:
	std::string param_path;
	std::string bin_path;
	std::vector<int> input_shape;
	ncnn::Net net;

	const float norm_vals[3] = { 1 / 177.5, 1 / 177.5, 1 / 177.5 };
	const float mean_vals[3] = { 175.5, 175.5, 175.5 };

	cv::Mat normalize(cv::Mat& image);
public:
	HumanSeg() = delete;
	HumanSeg(const std::string param_path, const std::string bin_path, std::vector<int> input_shape);
	~HumanSeg();
	
	cv::Mat predict_image(cv::Mat& image);
	void predict_image(const std::string& src_image_path, const std::string& dst_path);
	
	void predict_camera();
};

