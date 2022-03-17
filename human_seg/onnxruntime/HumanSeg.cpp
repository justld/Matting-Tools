#include "HumanSeg.h"


HumanSeg::HumanSeg(std::wstring model_path, int num_threads = 1, std::vector<int64_t> input_node_dims = { 1, 3, 192, 192 }) {
	input_node_dims_ = input_node_dims;
	for (int64_t i : input_node_dims_) {
		input_tensor_size_ *= i;
		out_tensor_size_ *= i;
	}

	std::cout << input_tensor_size_ << std::endl;
	session_options_.SetIntraOpNumThreads(num_threads);
	session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	try {
		session_ = Ort::Session(env_, model_path.c_str(), session_options_);
	}
	catch (Ort::Exception& e) {
		std::cout << e.what() << std::endl;
	}
	
	Ort::AllocatorWithDefaultOptions allocator;
	//»ñÈ¡ÊäÈëname
	const char* input_name = session_.GetInputName(0, allocator);
	input_node_names_ = { input_name };
	//std::cout << "input name:" << input_name << std::endl;
	const char* output_name = session_.GetOutputName(0, allocator);
	out_node_names_ = { output_name };
	//std::cout << "output name:" << output_name << std::endl;
}


cv::Mat HumanSeg::normalize(cv::Mat& image) {
	std::vector<cv::Mat> channels, normalized_image;
	cv::split(image, channels);

	cv::Mat r, g, b;
	b = channels.at(0);
	g = channels.at(1);
	r = channels.at(2);
	b = (b / 255. - 0.5) / 0.5;
	g = (g / 255. - 0.5) / 0.5;
	r = (r / 255. - 0.5) / 0.5;

	normalized_image.push_back(r);
	normalized_image.push_back(g);
	normalized_image.push_back(b);

	cv::Mat out = cv::Mat(image.rows, image.cols, CV_32F);
	cv::merge(normalized_image, out);
	return out;
}

/*
* preprocess: resize -> normalize
*/
cv::Mat HumanSeg::preprocess(cv::Mat image) {
	image_h = image.rows;
	image_w = image.cols;
	cv::Mat dst, dst_float, normalized_image;
	cv::resize(image, dst, cv::Size(int(input_node_dims_[3]), int(input_node_dims_[2])), 0, 0);
	dst.convertTo(dst_float, CV_32F);
	normalized_image = normalize(dst_float);
	
	return normalized_image;
}

/*
* postprocess: preprocessed image -> infer -> postprocess
*/
void HumanSeg::predict_image(const std::string& src_path, const std::string& dst_path) {
	cv::Mat image = cv::imread(src_path);
	cv::Mat preprocessed_image = preprocess(image);
	cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1, cv::Size(int(input_node_dims_[3]), int(input_node_dims_[2])), cv::Scalar(0, 0, 0), false, true);
	//std::cout << "load image success." << std::endl;
	// create input tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	input_tensors_.emplace_back( Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_node_dims_.data(), input_node_dims_.size()) );

	std::vector<Ort::Value> output_tensors_ = session_.Run(
		Ort::RunOptions{ nullptr }, 
		input_node_names_.data(),
		input_tensors_.data(), 
		input_node_names_.size(), 
		out_node_names_.data(),
		out_node_names_.size()
	);
	int64* floatarr = output_tensors_[0].GetTensorMutableData<int64>();

	// decoder 
	cv::Mat mask = cv::Mat::zeros(static_cast<int>(input_node_dims_[2]), static_cast<int>(input_node_dims_[3]), CV_8UC1);
	
	for (int i{ 0 }; i < static_cast<int>(input_node_dims_[2]); i++) {
		for (int j{ 0 }; j < static_cast<int>(input_node_dims_[3]); ++j) {
			mask.at<uchar>(i, j) = static_cast<uchar>(floatarr[i * static_cast<int>(input_node_dims_[2]) + j]);
		}
	}
	cv::resize(mask, mask, cv::Size(image_w, image_h), 0, 0);
	cv::Mat predict_image;
	cv::bitwise_and(image, image, predict_image, mask = mask);
	cv::imwrite(dst_path, predict_image);
	
	//std::cout << "predict image over" << std::endl;
	input_tensors_.clear();
}


