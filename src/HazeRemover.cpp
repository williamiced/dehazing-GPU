#include "HazeRemover.h"

using namespace std;
using namespace cv;

int HazeRemover::getLoopCount() {
	if (endsWith(mInputFilePath, ".png") || endsWith(mInputFilePath, ".jpg") || endsWith(mInputFilePath, ".jpeg")) {
		mIsVideo = false;
		return 1;
	} else {
		mVideoCapture = new VideoCapture(mInputFilePath);
		mIsVideo = true;
		mVideoWriter = new VideoWriter("Dehaze.avi", 
							CV_FOURCC('D', 'I', 'V', 'X'), 
							mVideoCapture->get(CV_CAP_PROP_FPS), 
							Size(mVideoCapture->get(CV_CAP_PROP_FRAME_WIDTH), mVideoCapture->get(CV_CAP_PROP_FRAME_HEIGHT))
						);

		return mVideoCapture->get(CV_CAP_PROP_FRAME_COUNT);
	}
}

void HazeRemover::loadImage() {
	if (mIsVideo) {
		bool isSuccess = false;
		do {
			isSuccess = mVideoCapture->read(mInputImg);
		} while(!isSuccess);
	} else {
		mInputImg = imread(mInputFilePath);
	}
	mInputImg.convertTo(mInputImg, CV_32FC3);	
}

void HazeRemover::postProcess() {
	if (mIsVideo)
		mVideoCapture->release();
	
	coreMemDestroy();
	guidedMemDestroy();
	softMattingMemDestroy();
}

void HazeRemover::gpuMemInit() {
	coreMemInit(mInputImg.cols, mInputImg.rows, mInputImg.channels());
	guidedMemInit();
	softMattingMemInit();
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

	if (mIsVideo) {
		dehazeImage.convertTo(dehazeImage, CV_8UC3);
		mVideoWriter->write(dehazeImage);
	}
	else {
		imwrite("Dehaze.png", dehazeImage);	
	}
}

void HazeRemover::dehaze() {
	int loopCount = getLoopCount();

	float* A = nullptr;
	bool isFirstLoop = true;

	while (loopCount > 0) {
		// load image from image or video
		loadImage();

		// Init GPU memory for the first loop
		if (isFirstLoop) {
			gpuMemInit();
			preCalcForSoftMatting();
			
			isFirstLoop = false;
		}
		
		setData((float*)mInputImg.data);

		// Calculate Dark Channel
		calcDarkChannel();
#ifdef __DEBUG
		saveDarkChannelImage();
#endif

		if (A == nullptr) {
			A = new float[mInputImg.channels() * sizeof(float)];

			// Calculate Air Light
			calcAirLight(A, (float*) mInputImg.data);
		}

		// Calculate Transmisison
		calcTransmission(A);

		// Refine Transmission using Soft-Matting
		doGuidedFilter();
#ifdef __DEBUG
		saveTransmissionImage();
#endif

		if (mIsVideo) {
			gRefineGPU = gTransPatchGPU;
		} else {
			refineTransmission();
#ifdef __DEBUG
			saveRefineImage();
#endif	
		}
		
		// Dehaze
		doDehaze(A);
		saveDehazeImage();	

		loopCount--;
	}
	
	if (A != nullptr)
		delete[] A;
	postProcess();
}

void HazeRemover::setInputFilePath(string filePath) {
	mInputFilePath = filePath;
}

HazeRemover::HazeRemover() {
	cout << "Construct HazeRemover" << endl;
}

HazeRemover::~HazeRemover() {
}
