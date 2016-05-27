#include "softmatting.h"

float* gN;
float* gA;
float* gB;
float* gMeanA;
float* gMeanB;
float* gMeanI;
float* gMeanP;
float* gII;
float* gIP;
float* gMeanII;
float* gMeanIP;

__global__ void getNMatrix(float* N, int width, int height, int window) {
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

__global__ void applyMeanFilter(float* I_1, float* I_2, float* N, float* O_1, float* O_2, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total_1 = 0.f;
		float total_2 = 0.f;
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) {
					total_1 += I_1[cy * width + cx];
					total_2 += I_2[cy * width + cx];
				}
			}
		}
		O_1[i] = total_1 / N[i];
		O_2[i] = total_2 / N[i];
	}
}

__global__ void getIIAndIPMatrix(float* I, float* P, float* II, float* IP, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		II[i] = I[i] * I[i];
		IP[i] = I[i] * P[i];
	}
}

__global__ void getAAndBMatrix(float* a, float* b, float* meanI, float* meanP, float* meanII, float* meanIP, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float covIP = meanIP[i] - meanI[i] * meanP[i];
		float varI =  meanII[i] - meanI[i] * meanI[i];
		a[i] = covIP / (varI + PARAM_EPSILON);
		b[i] = meanP[i] - a[i] * meanI[i];
	}
}

__global__ void getResultMatrix(float* I, float* meanA, float* meanB, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		I[i] = I[i] * meanA[i] + meanB[i];
	}
}

void initMemForSoftMatting() {
	CUDA_CHECK_RETURN( cudaMalloc(&gN, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gA, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gB, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gMeanA, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gMeanB, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gMeanI, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gMeanP, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gII, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gIP, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gMeanII, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gMeanIP, gImgWidth * gImgHeight * sizeof(float) ) );
}

void refineTransmission() {
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);

	initMemForSoftMatting();

	getNMatrix<<<gdim, bdim>>>(gN, gImgWidth, gImgHeight, WINDOW2);
	getIIAndIPMatrix<<<gdim, bdim>>>(gGrayGPU, gTransPatchGPU, gII, gIP, gImgWidth, gImgHeight);
	CHECK

	applyMeanFilter<<<gdim, bdim>>>(gGrayGPU, gTransPatchGPU, gN, gMeanI, gMeanP, gImgWidth, gImgHeight, WINDOW2);
	applyMeanFilter<<<gdim, bdim>>>(gII, gIP, gN, gMeanII, gMeanIP, gImgWidth, gImgHeight, WINDOW2);
	CHECK

	getAAndBMatrix<<<gdim, bdim>>>(gA, gB, gMeanI, gMeanP, gMeanII, gMeanIP, gImgWidth, gImgHeight);
	CHECK

	applyMeanFilter<<<gdim, bdim>>>(gA, gB, gN, gMeanA, gMeanB, gImgWidth, gImgHeight, WINDOW2);
	CHECK
	
	getResultMatrix<<<gdim, bdim>>>(gTransPatchGPU, gMeanA, gMeanB, gImgWidth, gImgHeight);
	CHECK
}

void fillRefineData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gTransPatchGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost) );
}