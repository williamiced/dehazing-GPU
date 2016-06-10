#include "guidedfilter.h"

float* 	gGuidedGPU;
float* 	gN;
float* 	gMeanI_g
float*  gMeanI_t;
float*  sigmai;
float*  cross;
float*  a;
float*  b;
float*  meanA;
float*  meanB;

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
		guidedResult[i] = meanA[i] * input[i] + meanB;
	}
}

void initMemForGuidedFilter() {
	// 1 * 1
	CUDA_CHECK_RETURN( cudaMalloc((void **) &gGuidedGPU, gImgWidth * gImgHeight * sizeof(float) ) );

	// 1 * 1
	CUDA_CHECK_RETURN( cudaMalloc((void **) &gN, gImgWidth * gImgHeight * sizeof(float) ) );

	// 3 * 1
	CUDA_CHECK_RETURN( cudaMalloc((void **) &gMeanI_g, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc((void **) &gMeanI_t, gImgWidth * gImgHeight * sizeof(float) ) );

	//
	CUDA_CHECK_RETURN( cudaMalloc((void **) &sigmai, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc((void **) &cross, gImgWidth * gImgHeight * sizeof(float) ) );

	CUDA_CHECK_RETURN( cudaMalloc((void **) &a, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc((void **) &b, gImgWidth * gImgHeight * sizeof(float) ) );

	CUDA_CHECK_RETURN( cudaMalloc((void **) &meanA, gImgWidth * gImgHeight * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc((void **) &meanB, gImgWidth * gImgHeight * sizeof(float) ) );

}

void guidedFilter(){
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);

	getNMatrix<<<gdim, bdim>>>(gN, gImgWidth, gImgHeight, WINDOW2);
	CHECK

	//use gray image as guided image
	getMeanMatrixSingleChannel<<<gdim, bdim>>>(gGrayGPU, gMeanI_g, gN, gImgWidth, gImgHeight, WINDOW2);
	CHECK

	getMeanMatrixSingleChannel<<<gdim, bdim>>>(gTransPatchGPU, gMeanI_t, gN, gImgWidth, gImgHeight, WINDOW2);
	CHECK

	getCross<<<gdim, bdim>>>(gGrayGPU, gTransPatchGPU, cross, gN, gImgWidth, gImgHeight, WINDOW2);
	CHECK

	getSigma<<<gdim, bdim>>>(gGrayGPU, sigmai, gN, gMeanI_g, gImgWidth, gImgHeight, WINDOW2);
	CHECK
	
	calculateLinearCoefficients<<<gdim, bdim>>>(cross, sigmai, gMeanI_g, gMeanI_t, a, b, gImgWidth, gImgHeight, WINDOW2); 
	CHECK

	getMeanMatrixSingleChannel<<<gdim, bdim>>>(a, meanA, gN, gImgWidth, gImgHeight, WINDOW2);
	getMeanMatrixSingleChannel<<<gdim, bdim>>>(b, meanB, gN, gImgWidth, gImgHeight, WINDOW2);
	CHECK

	filter<<<gdim, bdim>>>(meanA, meanB, gGuidedGPU, gGrayGPU ,gImgWidth, gImgHeight, WINDOW2);
	CHECK

}

void fillGuidedData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gGuidedGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost) );
}	