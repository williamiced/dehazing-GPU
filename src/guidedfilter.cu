#include "guidedfilter.h"

float* 	gN_g;
float* 	gMeanI_g;
float*  gMeanI_t;
float*  gSigmai;
float*  gCross;
float*  gA;
float*  gB;
float*  gMeanA;
float*  gMeanB;

__global__ void getNMatrix2(float* N, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total = 0.f;
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) 
					total += 1.f;
			}
		}
		N[i] = total;
	}
}

__global__ void getMeanMatrixSingleChannel(float* I, float* meanI, float* N, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total = 0.f;
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) {
					int newI = cy * width + cx;
					total += I[newI];
				}
			}
		}
		meanI[i] = total / N[i];
	}
}

__global__ void getCross(float* Guided, float* transmission, float* cross, float* N, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total = 0.f;
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) {
					int newI = cy * width + cx;
					total += Guided[newI] * transmission[newI];
				}
			}
		}
		cross[i] = total / N[i];
	}
}

__global__ void getSigma(float* Guided, float* sigmai, float* N, float* mean, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total = 0.f;
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) {
					int newI = cy * width + cx;
					total += Guided[newI] * Guided[newI];
				}
			}
		}
		sigmai[i] = total / N[i] - mean[i] * mean[i];
	}
}


__global__ void calculateLinearCoefficients(float* cross, float* sigmai, float* meanG, float* meanT, float* a, float* b, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		a[i] = (cross[i] - meanG[i] * meanT[i]) / (sigmai[i] + PARAM_EPSILON);
		b[i] = meanT[i] - a[i] * meanG[i];
	}
}

__global__ void filter(float* meanA, float* meanB, float* guidedResult, float* input, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		guidedResult[i] = meanA[i] * input[i] + meanB[i];
	}
}

void guidedMemInit() {
	gN_g 		= g1ChannelContainerGPU[2];
	gSigmai 	= g1ChannelContainerGPU[3];
	gCross 		= g1ChannelContainerGPU[4];
	gA 			= g1ChannelContainerGPU[5];
	gB 			= g1ChannelContainerGPU[6];
	gMeanI_g	= g1ChannelContainerGPU[7];
	gMeanI_t	= g1ChannelContainerGPU[8];
	gMeanA 		= g1ChannelContainerGPU[7];
	gMeanB 		= g1ChannelContainerGPU[8];
}

void guidedMemDestroy() {
}

void doGuidedFilter(){
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);

	getNMatrix2<<<gdim, bdim>>>(gN_g, gImgWidth, gImgHeight, WINDOW_GF);
	CHECK

	//use gray image as guided image
	getMeanMatrixSingleChannel<<<gdim, bdim>>>(gGrayGPU, gMeanI_g, gN_g, gImgWidth, gImgHeight, WINDOW_GF);
	CHECK

	getMeanMatrixSingleChannel<<<gdim, bdim>>>(gTransPatchGPU, gMeanI_t, gN_g, gImgWidth, gImgHeight, WINDOW_GF);
	CHECK

	getCross<<<gdim, bdim>>>(gGrayGPU, gTransPatchGPU, gCross, gN_g, gImgWidth, gImgHeight, WINDOW_GF);
	CHECK

	getSigma<<<gdim, bdim>>>(gGrayGPU, gSigmai, gN_g, gMeanI_g, gImgWidth, gImgHeight, WINDOW_GF);
	CHECK
	
	calculateLinearCoefficients<<<gdim, bdim>>>(gCross, gSigmai, gMeanI_g, gMeanI_t, gA, gB, gImgWidth, gImgHeight, WINDOW_GF); 
	CHECK

	getMeanMatrixSingleChannel<<<gdim, bdim>>>(gA, gMeanA, gN_g, gImgWidth, gImgHeight, WINDOW_GF);
	getMeanMatrixSingleChannel<<<gdim, bdim>>>(gB, gMeanB, gN_g, gImgWidth, gImgHeight, WINDOW_GF);
	CHECK

	filter<<<gdim, bdim>>>(gMeanA, gMeanB, gTransPatchGPU, gGrayGPU ,gImgWidth, gImgHeight, WINDOW_GF);
	CHECK
}

void fillGuidedData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gTransPatchGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost) );
}	