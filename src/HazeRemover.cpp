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

void HazeRemover::saveTransmissionImage() {
	Mat transmissionImage = Mat::zeros(mInputImg.size(), CV_32FC1);
	fillTransmissionData((float*) transmissionImage.data);
	transmissionImage *= 255.f;
	imwrite("Transmission.png", transmissionImage);	
}

void HazeRemover::saveRefineImage() {
	Mat refineImage = Mat::zeros(mInputImg.size(), CV_32FC1);
	fillRefineData((float*) refineImage.data);
	refineImage *= 255.f;
	imwrite("Refine.png", refineImage);
}

void HazeRemover::saveDehazeImage() {
	Mat dehazeImage = Mat::zeros(mInputImg.size(), CV_32FC3);
	fillDehazeData((float*) dehazeImage.data);
	imwrite("Dehaze.png", dehazeImage);	
}

void HazeRemover::dehaze() {	
	// Pre-process
	preProcess();
	
	// Calculate Dark Channel
	calcDarkChannel();
	saveDarkChannelImage();

	// Calculate Air Light
	float* A = new float[mInputImg.channels() * sizeof(float)];
	calcAirLight(A, (float*) mInputImg.data);

	// Calculate Transmisison
	calcTransmission(A);
	saveTransmissionImage();

	// Refine Transmission using Soft-Matting
	refineTransmission();
	saveRefineImage();

	// Dehaze
	doDehaze(A);
	saveDehazeImage();

	delete[] A;
	gpuMemDestroy();
}

void HazeRemover::setInputFilePath(string filePath) {
	mInputFilePath = filePath;
}

HazeRemover::HazeRemover() {
	cout << "Construct HazeRemover" << endl;
}

HazeRemover::~HazeRemover() {
}