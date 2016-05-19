#ifndef _H_HAZE_REMOVER
#define _H_HAZE_REMOVER

#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "dehazing.h"

class HazeRemover {
private:
	void preProcess();
	void loadImage();
	void saveDarkChannelImage();

	std::string mInputFilePath;
	cv::Mat 	mInputImg;

public:
	void dehaze();
	void setInputFilePath(std::string filePath);
	HazeRemover();
};

#endif // _H_HAZE_REMOVER