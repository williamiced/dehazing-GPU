#ifndef _H_HAZE_REMOVER
#define _H_HAZE_REMOVER

#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "dehazing.h"
#include "guidedfilter.h"
#include "softmatting.h"

class HazeRemover {
private:
	void postProcess();
	int getLoopCount();
	void loadImage();
	void gpuMemInit();
	void saveDarkChannelImage();
	void saveTransmissionImage();
	void saveRefineImage();
	void saveDehazeImage();

	std::string mInputFilePath;
	cv::Mat 	mInputImg;
	cv::VideoCapture* mVideoCapture;
	bool		mIsVideo;

public:
	void dehaze();
	void setInputFilePath(std::string filePath);

	HazeRemover();
	~HazeRemover();
};

inline bool endsWith(std::string const & value, std::string const & ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

#endif // _H_HAZE_REMOVER
