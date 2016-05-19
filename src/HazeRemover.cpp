#include "HazeRemover.h"

using namespace std;
using namespace cv;

void HazeRemover::loadImage() {
	mInputImg = imread(mInputFilePath);
	mInputImg.convertTo(mInputImg, CV_32FC3);
}

void HazeRemover::preProcess() {
	loadImage();

	// GPU memory allocation
	gpuMemInit(mInputImg.cols, mInputImg.rows, mInputImg.channels(), (float*) mInputImg.data);
}

void HazeRemover::saveDarkChannelImage() {
	Mat darkChannelImage = Mat::zeros(mInputImg.size(), CV_32FC1);
	fillDarkChannelData((float*) darkChannelImage.data);
	imwrite("DarkChannel.png", darkChannelImage);
}

void HazeRemover::dehaze() {	
	// Pre-process
	preProcess();
	calcDarkChannel();
	saveDarkChannelImage();
}

void HazeRemover::setInputFilePath(string filePath) {
	mInputFilePath = filePath;
}

HazeRemover::HazeRemover() {
	cout << "Construct HazeRemover" << endl;
}