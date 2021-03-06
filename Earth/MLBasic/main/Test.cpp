#pragma once
#ifndef TEST_H
#define TEST_H

#include <iostream>
 
#include "../../MLBasic/test/TestImageUtil.h"
#include "../../MLBasic/test/TestDenseSIFT.h"
#include "../../MLBasic/test/TestUtil.h"
#include "../../MLBasic/test/TestVocabulary.h"
#include "../../MLBasic/test/TestHist.h"
#include "../../MLBasic/test/TestUtilWriteMat.h"

using namespace std;
using namespace cv;

int main() {
	/*fprintf(stderr, "....................start.....................\n");
	int ImageUtilSucc = TestImageUtil::test();
	if (ImageUtilSucc != 0) {
		fprintf(stderr, "ImageUtil test fail.\n");
		return -1;
	}
	fprintf(stderr, "ImageUtil test success.\n");
	fprintf(stderr, "..............................................\n");
	int DenseSIFTSucc = TestDenseSIFT::test();
	if (DenseSIFTSucc != 0) {
		fprintf(stderr, "DenseSIFT test fail.\n");
		return -1;
	}
	fprintf(stderr, "DenseSIFT test success.\n");
	fprintf(stderr, "..................finished....................\n");*/

	//TestUtil::test();

	//TestHist::test();
	TestUtilWriteMat::test();



	return 0;
}

#endif