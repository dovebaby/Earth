#pragma once
#ifndef DENSE_SIFT_H
#define DENSE_SIFT_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#ifdef _DEBUG
#pragma comment(lib, "opencv_core2410d")
#pragma comment(lib, "opencv_highgui2410d")
#pragma comment(lib, "opencv_features2d2410d")
#pragma comment(lib, "opencv_ml2410d")
#pragma comment(lib, "opencv_nonfree2410d")
#pragma comment(lib, "opencv_imgproc2410d")
#else
#pragma comment(lib, "opencv_core2410")
#pragma comment(lib, "opencv_highgui2410")
#pragma comment(lib, "opencv_features2d2410")
#pragma comment(lib, "opencv_ml2410")
#pragma comment(lib, "opencv_nonfree2410")
#pragma comment(lib, "opencv_imgproc2410")
#endif

class DenseSIFTFeatureExtend{
private:
	Ptr<DenseFeatureDetector> feature_detector;
	Ptr<DescriptorExtractor> descriptor_extractor;
	bool scale;
	int weigth;
	int heigth;
	// resize to 256x256
	cv::Mat UnifyImageSize(const cv::Mat& image)
	{
		Mat unified_image;
		Size size;
		if (scale) {
			size = Size(weigth, heigth);
		} else {
			int s = cv::min(image.rows, image.cols);
			float scale = weigth / s;
			size = Size(image.cols  * scale, image.rows * scale);
		}
		cv::resize(image, unified_image, size);
		cv::medianBlur(unified_image, unified_image, 3);
		return unified_image;
	}
	
	DenseSIFTFeatureExtend(int minHessian){
		//Ptr<FeatureDetector> feature_detector(new SurfFeatureDetector(minHessian));
		feature_detector = new DenseFeatureDetector();
		//this.descriptor_extractor = DescriptorExtractor::create("GridFAST");
		descriptor_extractor = new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SurfDescriptorExtractor(minHessian)));
		//this.descriptor_extractor = new SurfDescriptorExtractor();
	}

public:
	
	static DenseSIFTFeatureExtend build(int weight = 256, int height = 256, int minHessian = 400, bool scale = false) {
		DenseSIFTFeatureExtend res = DenseSIFTFeatureExtend(minHessian);
		return res;
	}

	pair<vector<cv::KeyPoint>, cv::Mat> extract(const cv::Mat& image) {
		cv::Mat new_image = UnifyImageSize(image);
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		feature_detector->detect(new_image, keypoints);
        descriptor_extractor->compute(new_image, keypoints, descriptors);
		return make_pair(keypoints, descriptors);
	}
};

#endif