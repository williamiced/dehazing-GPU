#include "dehazing.h"
#include <stdio.h>
#include <iostream>
//convenient macros
#define IN_GRAPH(x,y,w,h) ((x>=0)&&(x<w)&&(y>=0)&&(y<h))
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)
#define WINDOW 7
#define R 15

float* 	gImgGPU;
float* 	gDarkPixelGPU;
float* 	gDarkPatchGPU;
float* 	gGrayGPU;
int 	gImgWidth;
int 	gImgHeight;
int 	gImgChannels;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void kernelDarkPixel(float3 *image, float *imgGrey, float *dark, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		dark[i] = min(image[i].x, min(image[i].y, image[i].z));
		imgGrey[i] = image[i].x * 0.299 +  image[i].y * 0.587 + image[i].z * 0.114;
	}
}

//second kernel calculate min of 15 X 15 ceil
__global__ void kernelDarkPatch(float *dark, float *new_dark, int width, int height, int window) {
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

void gpuMemInit(int width, int height, int channels, float* rawData) {
	gImgWidth = width;
	gImgHeight = height;
	gImgChannels = channels;

	CUDA_CHECK_RETURN( cudaMalloc(&gImgGPU, width * height * channels * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gDarkPixelGPU, width * height * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gDarkPatchGPU, width * height * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gGrayGPU, width * height * channels * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMemcpy(gImgGPU, rawData, width * height * channels * sizeof(float), cudaMemcpyHostToDevice));
}

void calcDarkChannel() {
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);
	
	kernelDarkPixel<<<gdim, bdim>>> ((float3*) gImgGPU, gGrayGPU, gDarkPixelGPU, gImgWidth, gImgHeight);
	kernelDarkPatch<<<gdim, bdim>>> (gDarkPixelGPU, gDarkPatchGPU, gImgWidth, gImgHeight, WINDOW);
}

void fillDarkChannelData(float* cpuData) {
	cudaMemcpy(cpuData, gDarkPatchGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost);
}