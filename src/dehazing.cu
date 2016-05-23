#include "dehazing.h"
#include <stdio.h>
#include <iostream>
//convenient macros
#define IN_GRAPH(x,y,w,h) ((x>=0)&&(x<w)&&(y>=0)&&(y<h))
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)
#define WINDOW 7
#define PARAM_OMEGA 0.95
#define PARAM_T0 0.1

using namespace std;

float* 	gImgGPU;
float* 	gDarkPixelGPU;
float* 	gDarkPatchGPU;
float*	gTransPixelGPU;
float*	gTransPatchGPU;
float* 	gGrayGPU;
int 	gImgWidth;
int 	gImgHeight;
int 	gImgChannels;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void kernelDarkPixel(float3* image, float* imgGrey, float* dark, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		dark[i] = min(image[i].x, min(image[i].y, image[i].z));
		imgGrey[i] = image[i].x * 0.299 +  image[i].y * 0.587 + image[i].z * 0.114;
	}
}

//second kernel calculate min of 15 X 15 ceil
__global__ void kernelDarkPatch(float* dark, float* new_dark, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height){
		float minval = 255.0;
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) {
					minval = min(minval, dark[cy * width + cx]);
				}
			}
		}
		new_dark[i] = minval;
	}
}

// Max reduction for every 1024 pixels
__global__ void kernelMaxReduction(float* dark, int size, unsigned int* selectedIdx) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	extern __shared__ float sharedDark[];
	unsigned int* sharedIdx = (unsigned int*) (sharedDark + blockDim.x);

	if(i < size) {
		sharedDark[threadIdx.x] = dark[i];
		sharedIdx[threadIdx.x] = i;
		__syncthreads();

		for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
			if(threadIdx.x < stride){
				if(sharedDark[threadIdx.x + stride] > sharedDark[threadIdx.x]) {
					sharedDark[threadIdx.x] = sharedDark[threadIdx.x + stride];
					sharedIdx[threadIdx.x] = sharedIdx[threadIdx.x + stride];
				}
			}
			__syncthreads();
		}
		if(threadIdx.x == 0) 
			selectedIdx[blockIdx.x] = sharedIdx[0];
	} 
}

//calculate air light
__global__ void kernelGetTopIntensity(float* gray, int size, unsigned int* selectedIdx) {
	extern __shared__ float sharedGray[];
	unsigned int* sharedIdx = (unsigned int*) (sharedGray + blockDim.x);
	
	sharedGray[threadIdx.x] = gray[selectedIdx[threadIdx.x]];
	sharedIdx[threadIdx.x] = selectedIdx[threadIdx.x];
	__syncthreads();

	for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
		if(threadIdx.x < stride){
			if(sharedGray[threadIdx.x + stride] > sharedGray[threadIdx.x]) {
				sharedGray[threadIdx.x] = sharedGray[threadIdx.x + stride];
				sharedIdx[threadIdx.x] = sharedIdx[threadIdx.x + stride];
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) 
		selectedIdx[0] = sharedIdx[0];
	// Result idx is on selectedIdx
}

__global__ void kernelNormalizeByAirLight (float3* img, float* trans, float nx, float ny, float nz, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	float tx, ty, tz;
	if(x < width && y < height) {
		tx = img[i].x / nx;
		ty = img[i].y / ny;
		tz = img[i].z / nz;
		trans[i] = min(tx, min(ty, tz));
	}
}

__global__ void kernelTransPatch (float* trans, float* new_trans, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float minval = 255.0;
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) {
					minval = min(minval, trans[cy * width + cx]);
				}
			}
		}
		new_trans[i] = 1.0 - PARAM_OMEGA * minval;
	}
}

__global__ void kernelDoDehaze (float3* img, float* trans, float Ax, float Ay, float Az, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float denominator = max(PARAM_T0, trans[i]);
		img[i].x = (img[i].x - Ax) / denominator + Ax;
		img[i].y = (img[i].y - Ay) / denominator + Ay;
		img[i].z = (img[i].z - Az) / denominator + Az;
	}
}

void gpuMemInit(int width, int height, int channels, float* rawData) {
	gImgWidth = width;
	gImgHeight = height;
	gImgChannels = channels;

	CUDA_CHECK_RETURN( cudaMalloc(&gImgGPU, width * height * channels * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gDarkPixelGPU, width * height * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gDarkPatchGPU, width * height * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gTransPixelGPU, width * height * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gTransPatchGPU, width * height * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gGrayGPU, width * height * channels * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMemcpy(gImgGPU, rawData, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));

	CHECK
}

void gpuMemDestroy() {
	CUDA_CHECK_RETURN( cudaFree(gImgGPU) );
	CUDA_CHECK_RETURN( cudaFree(gDarkPatchGPU) );
	CUDA_CHECK_RETURN( cudaFree(gGrayGPU) );

	CHECK
}

void calcDarkChannel() {
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);
	
	kernelDarkPixel<<<gdim, bdim>>> ((float3*) gImgGPU, gGrayGPU, gDarkPixelGPU, gImgWidth, gImgHeight);
	CHECK

	kernelDarkPatch<<<gdim, bdim>>> (gDarkPixelGPU, gDarkPatchGPU, gImgWidth, gImgHeight, WINDOW);
	CHECK

	// No need anymore
	CUDA_CHECK_RETURN( cudaFree(gDarkPixelGPU) );
}

void calcAirLight(float* A, float* rawData) {
	SETUP_TIMER
	
	dim3 bdim(1024);
	dim3 gdim(CEIL(double(gImgWidth * gImgHeight) / bdim.x));
	
	unsigned int* selectedIdx = NULL;
	CUDA_CHECK_RETURN( cudaMalloc((void **)(&selectedIdx), sizeof(unsigned int) * gdim.x) );
	
	int sharedSize1 = bdim.x * (sizeof(float) + sizeof(unsigned int));
	int sharedSize2 = gdim.x * (sizeof(float) + sizeof(unsigned int));
	
	kernelMaxReduction<<<gdim, bdim, sharedSize1>>> (gDarkPatchGPU, gImgWidth * gImgHeight, selectedIdx);
	CHECK

	kernelGetTopIntensity<<<1, gdim, sharedSize2>>> (gGrayGPU, gImgWidth * gImgHeight, selectedIdx);
	CHECK

	unsigned int resultIdx;
	CUDA_CHECK_RETURN( cudaMemcpy(&resultIdx, selectedIdx, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

	for (int i=0; i<gImgChannels; i++) 
		A[i] = rawData[resultIdx * gImgChannels + i];

	CUDA_CHECK_RETURN( cudaFree(selectedIdx) );
}

void calcTransmission(float* A) {
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);

	kernelNormalizeByAirLight<<<gdim, bdim>>> ((float3*) gImgGPU, gTransPixelGPU, A[0], A[1], A[2], gImgWidth, gImgHeight);
	CHECK

	kernelTransPatch<<<gdim, bdim>>> (gTransPixelGPU, gTransPatchGPU, gImgWidth, gImgHeight, WINDOW);
	CHECK

	// No need anymore
	CUDA_CHECK_RETURN( cudaFree(gTransPixelGPU) );
}

void doDehaze(float* A) {
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);

	kernelDoDehaze<<<gdim, bdim>>>( (float3*)gImgGPU, gTransPatchGPU, A[0], A[1], A[2], gImgWidth, gImgHeight);
}

void fillDarkChannelData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gDarkPatchGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost) );
}

void fillTransmissionData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gTransPatchGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost) );
}

void fillDehazeData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gImgGPU, gImgWidth * gImgHeight * gImgChannels * sizeof(float), cudaMemcpyDeviceToHost) );
}