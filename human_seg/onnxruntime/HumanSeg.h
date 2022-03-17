#pragma once
#include <string>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


class HumanSeg
{
protected:
	Ort::Env env_;
	Ort::SessionOptions session_options_;
	Ort::Session session_{nullptr};
	Ort::RunOptions run_options_{nullptr};

	std::vector<Ort::Value> input_tensors_;


	std::vector<const char*> input_node_names_;
	std::vector<int64_t> input_node_dims_;
	size_t input_tensor_size_{ 1 };

	std::vector<const char*> out_node_names_;
	size_t out_tensor_size_{ 1 };

	int image_h;
	int image_w;

	cv::Mat normalize(cv::Mat& image);
	cv::Mat preprocess(cv::Mat image);

public:
	HumanSeg() =delete;
	HumanSeg(std::wstring model_path, int num_threads, std::vector<int64_t> input_node_dims);
	HumanSeg(std::wstring model_path, int num_threads) {
		HumanSeg(model_path, num_threads, { 1, 3, 192, 192 });
	};
	void predict_image(const std::string& src_path, const std::string& dst_path);
	
	
};

