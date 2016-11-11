#include <iostream>
 
#include "../util/Util.h"
using namespace std;
using namespace cv;

class TestUtilWriteMat{
public:
	static int test() {

		
		string fileName = Util::getRealPath() + "\\data\\labels\\TestUtilWriteMat.txt";
		string image_file_path = Util::getRealPath() + "\\data\\images\\IMG_20161103_181500.jpg";
		Mat image = ImageUtil::load(image_file_path);
		if(image.empty())
		{
			fprintf(stderr, "ImageData.load is error\n");
			return -1;
		}
		DenseSIFT extract = DenseSIFT::build();
		extract.extractDescriptors(image);
		Mat descriptors = extract.getDescriptors();

		
		Util::writeMat(descriptors, fileName);




		return 0;
	}
};