#pragma once
#ifndef CONSTRUCT_HIST_H
#define CONSTRUCT_HIST_H


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

#include "../../../../../MLBasic/lib/cluster/kmeans.h"
#include "../../../../../MLBasic/lib/image/feature/bow/DenseSIFT.h"

class ConstructHist {
private:
	
	Mat vocabulary;
	vector<KeyPoint> keypoints;
	Mat imgDescriptor;
	DenseSIFT sift;
	Ptr<BOWImgDescriptorExtractor> hist;

	ConstructHist(DenseSIFT _sift):sift(_sift) {
		Ptr<DescriptorExtractor> dextractor = sift.getPtrDescriptorExtractor();
		Ptr<DescriptorMatcher> dmatcher = sift.getPtrDescriptorMatcher();
		hist = new BOWImgDescriptorExtractor(dextractor, dmatcher);
	}

public:

	void setLocalVocabulary(const Mat &_Vocabulary) {
		vocabulary = _Vocabulary;
		//hist->setVocabulary(vocabulary);
		hist->setVocabulary(vocabulary);
		/*hist;
		hist.obj;
		hist.obj->setVocabulary(vocabulary)*/;
		/*hist->
			setVocabulary(vocabulary);*/
		
		
	}

	Mat getVocabulary() {
		return vocabulary;
	}

	void computeImageDescriptor(const Mat & image) {
		Ptr<DenseFeatureDetector> dextractor = sift.getPtrDenseFeatureDetector();
		Mat std_image = sift.getSTDImage(image);
		dextractor->detect(std_image, keypoints);
		Mat tmpImgDescriptor;
		hist->compute(std_image, keypoints, tmpImgDescriptor);
		imgDescriptor = tmpImgDescriptor;
	}

	Mat getHistDescriptor() {
		return imgDescriptor;
	}

	vector<KeyPoint> getKeyPoints() {
		return keypoints;
	}

	static ConstructHist build(const DenseSIFT &sift, const Mat &vocabulary) {
		
		ConstructHist res = ConstructHist(sift);
		
		res.setLocalVocabulary(vocabulary);
		
		return res;
	}
};



#endif