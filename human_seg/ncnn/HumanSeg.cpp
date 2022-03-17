#include "HumanSeg.h"


HumanSeg::HumanSeg(const std::string param_path, const std::string bin_path, std::vector<int> input_shape)
	:param_path(param_path), bin_path(bin_path), input_shape(input_shape) {
	net.load_param(param_path.c_str());
	net.load_model(bin_path.c_str());
}


HumanSeg::~HumanSeg() {
	net.clear();
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


cv::Mat HumanSeg::predict_image(cv::Mat& image) {
	cv::Mat rgbImage;
	cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgbImage.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, input_shape[3], input_shape[2]);
	in.substract_mean_normalize(mean_vals, norm_vals);
	ncnn::Extractor ex = net.create_extractor();
	ex.input("x", in);
	ncnn::Mat out;
	ex.extract("bilinear_interp_v2_13.tmp_0", out);

	cv::Mat mask(out.h, out.w, CV_8UC1);
	const float* maskmap0 = out.channel(0);
	const float* maskmap1 = out.channel(1);

	for (int i{ 0 }; i < out.h; i++) {
		for (int j{ 0 }; j < out.w; ++j) {
			mask.at<uchar>(i, j) = maskmap1[i * out.w + j] > maskmap0[i * out.w + j] ? 255 : 0;
		}
	}
	cv::resize(mask, mask, cv::Size(image.cols, image.rows), 0, 0);
	cv::Mat segFrame;
	cv::bitwise_and(image, image, segFrame, mask = mask);
	return segFrame;
}


void HumanSeg::predict_image(const std::string& src_image_path, const std::string& dst_path) {
	cv::Mat image = cv::imread(src_image_path);
	cv::Mat segFrame = predict_image(image);
	cv::imwrite(dst_path, segFrame);
}


void HumanSeg::predict_camera() {
	cv::Mat frame;
	cv::VideoCapture cap;
	int deviceID{ 0 };
	int apiID{ cv::CAP_ANY };
	cap.open(deviceID, apiID);
	if (!cap.isOpened()) {
		std::cout << "Error, cannot open camera!" << std::endl;
		return;
	}
	//--- GRAB AND WRITE LOOP
	std::cout << "Start grabbing" << std::endl << "Press any key to terminate" << std::endl;
	int count{ 0 };
	clock_t start{ clock() }, end{0};
	double fps{ 0 };
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			std::cout << "ERROR! blank frame grabbed" << std::endl;
			break;
		}
		cv::Mat segFrame = predict_image(frame);
		
		// fps
		++count;
		end = clock();
		fps = count / (float(end - start) / CLOCKS_PER_SEC);
		if (count >= 50) {
			count = 0;  //防止计数溢出
			start = clock();
		}
		std::cout << "FPS: " << fps << "  Seg Image Number: " << count << "   time consume:" << (float(end - start) / CLOCKS_PER_SEC) << std::endl;
		//设置绘制文本的相关参数
		std::string text{ std::to_string(fps) };
		int font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 1;
		int thickness = 2;
		int baseline;
		cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

		//将文本框居中绘制
		cv::Point origin;
		origin.x = 20;
		origin.y = 20;
		cv::putText(segFrame, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

		// show live and wait for a key with timeout long enough to show images
		imshow("Live", segFrame);
		if (cv::waitKey(5) >= 0)
			break;

	}
	cap.release();
	cv::destroyWindow("Live");
	return;
}

